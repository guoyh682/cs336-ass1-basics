import argparse
import torch
import logging
import os
import json
import csv
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from train_bpe import train_bpe
from tokenizer import Tokenizer
from module import TransformerLM
from nn_utils import (
    AdamW,
    load_checkpoint,
    save_checkpoint,
    load_data,
    gradient_clipping,
    cosine_lr_schedule,
    cross_entropy,
)
import time
import math
from datetime import datetime
from pathlib import Path

def setup_logging_and_dirs(args):
    base_dir = Path("experiments")
    base_dir.mkdir(exist_ok=True)

    # 找到已有的最大编号
    existing = [int(p.name) for p in base_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    next_id = max(existing) + 1 if existing else 1
    experiment_dir = base_dir / f"{next_id:03d}"  # 3位编号，比如 001, 002
    checkpoints_dir = experiment_dir / "checkpoints"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(exist_ok=True)

    # 保存配置
    with open(experiment_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # 设置日志
    log_file = experiment_dir / "train.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()

    # 创建 CSV 文件记录指标
    metrics_file = experiment_dir / "metrics.csv"
    with open(metrics_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "val_loss", "perplexity", "learning_rate", "elapsed_time"])

    return logger, checkpoints_dir, metrics_file, experiment_dir

def encode_and_save(text_path, bin_file, tokenizer):
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()
    ids = tokenizer.encode(text)
    np.array(ids, dtype=np.uint16).tofile(bin_file)
    print(f"Saved {len(ids):,} tokens to {bin_file}")

if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Train a Transformer Language Model.")

    # Data and Checkpointing
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to the training data.")
    parser.add_argument("--val_data_path", type=str, required=True, help="Path to the validation data.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints.")

    # Model Hyperparameters
    parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size.")
    parser.add_argument("--context_length", type=int, default=256, help="Maximum context length for the model.")
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension.")
    parser.add_argument("--d_ff", type=int, default=1344, help="Dimension of the inner feed-forward layer.")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of Transformer layers.")
    parser.add_argument("--num_heads", type=int, default=16, help="Number of attention heads.")
    parser.add_argument("--theta", type=float, default=10000.0, help="Theta value for RoPE positional embeddings.")

    # Training Hyperparameters
    parser.add_argument("--max_learning_rate", type=float, default=1e-3, help="Maximum learning rate.")
    parser.add_argument("--min_learning_rate", type=float, default=1e-4, help="Minimum learning rate.")
    parser.add_argument("--beta1", type=float, default=0.9, help="AdamW beta1.")
    parser.add_argument("--beta2", type=float, default=0.95, help="AdamW beta2.")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay.")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value.")
    parser.add_argument("--warmup_iters", type=int, default=1000, help="Number of warmup iterations for LR scheduler.")
    parser.add_argument("--cosine_iters", type=int, default=None, help="Number of iterations for the cosine decay cycle.")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--max_steps", type=int, default=10000, help="Total number of training steps.")
    parser.add_argument("--eval_interval", type=int, default=500, help="Interval for evaluation and logging.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help='Device to train on (e.g., "cpu", "cuda", "mps").',
    )

    args = parser.parse_args()

    if args.cosine_iters is None:
        args.cosine_iters = args.max_steps

    logger, checkpoints_dir, metrics_file, experiment_dir = setup_logging_and_dirs(args)

    if args.train_data_path.endswith(".txt"):
        ## BPE Training
        print("--- BPE Training ---")
        vocab, merges = train_bpe(
            input_path=args.train_data_path, 
            vocab_size=args.vocab_size, 
            special_tokens=["<|endoftext|>"]
            )
        print("---------------------\n")
        ## Tokenization
        print("--- Tokenization ---")
        tokenizer = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])
        train_bin = "data/TinyStoriesTrain.bin"
        val_bin = "data/TinyStoriesVal.bin"
        encode_and_save(args.train_data_path, train_bin, tokenizer)
        encode_and_save(args.val_data_path, val_bin, tokenizer)
    else: 
        train_bin = args.train_data_path
        val_bin = args.val_data_path

    logger.info("--- Configuration ---")
    for k, v in vars(args).items():
        logger.info(f"{k}: {v}")
    logger.info("---------------------")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = args.device

    # --- Memory-efficient Data Loading ---
    print("Loading data...")
    train_data = np.memmap(train_bin, dtype=np.uint16, mode="r").astype(np.int64)
    val_data = np.memmap(val_bin, dtype=np.uint16, mode="r").astype(np.int64)
    print(f"Train data loaded with {len(train_data):,} tokens.")
    print(f"Validation data loaded with {len(val_data):,} tokens.\n")

    # --- Model and Optimizer Initialization ---
    print("Initializing model and optimizer...")
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.theta,
    ).to(device)

    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs")
    #     model = torch.nn.DataParallel(model)

    optimizer = AdamW(
        model.parameters(),
        lr=args.max_learning_rate,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )
    print(f"Model has {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters.\n")

    # --- Checkpoint Loading ---
    start_iter = 0
    checkpoint_path = checkpoints_dir / "latest_checkpoint.pth"
    if checkpoint_path.exists():
        try:
            start_iter = load_checkpoint(checkpoint_path, model, optimizer)
        except Exception as e:
            print(f"Could not load checkpoint. Starting from scratch. Error: {e}")

    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    # --- Training Loop ---
    print("Starting training loop...")
    start_time = time.time()
    train_losses = []
    val_losses = []

    for it in tqdm(range(start_iter, args.max_steps), desc="Training"):
        # --- Learning Rate Scheduling ---
        lr = cosine_lr_schedule(
            it, args.max_learning_rate, args.min_learning_rate, args.warmup_iters, args.cosine_iters
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # --- Evaluation ---
        if it % args.eval_interval == 0 or it == args.max_steps - 1:
            model.eval()
            val_loss_accum = 0.0
            eval_iters = 100
            with torch.no_grad():
                for _ in range(eval_iters):
                    x, y = load_data(val_data, args.batch_size, args.context_length, device)
                    logits = model(x)
                    loss = cross_entropy(logits, y)
                    val_loss_accum += loss.item()

            avg_val_loss = val_loss_accum / eval_iters
            val_losses.append(avg_val_loss)
            perplexity = math.exp(avg_val_loss)

            current_time = time.time()
            elapsed_time = current_time - start_time

            logger.info(
                f"Step {it} | Val Loss: {avg_val_loss:.4f} | Perplexity: {perplexity:.2f} "
                f"| LR: {lr:.6f} | Time: {elapsed_time:.2f}s"
            )
            tqdm.write(f"Step {it} | Loss: {loss.item():.4f}")

            # 写入 CSV
            with open(metrics_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([it, avg_val_loss, perplexity, lr, elapsed_time])

            save_checkpoint(model, optimizer, it, checkpoint_path)
            model.train()

        x, y = load_data(train_data, args.batch_size, args.context_length, device)
        logits = model(x)
        loss = cross_entropy(logits, y)
        train_losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), args.grad_clip)
        optimizer.step()

    print("\nTraining finished.")
    final_checkpoint_path = checkpoints_dir / f"final_model_step_{args.max_steps}.pth"
    save_checkpoint(model, optimizer, args.max_steps, final_checkpoint_path)
    print(f"Final model saved to {final_checkpoint_path}")

    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(range(0, args.max_steps+args.eval_interval, args.eval_interval), val_losses, label="Val Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(experiment_dir, "loss_curve.png")
    plt.savefig(save_path)
    print(f"Loss curve saved to {save_path}")
    plt.close()