import torch
import argparse
import torch.nn as nn
from module import softmax, TransformerLM
from tokenizer import Tokenizer

def generate(
    model: nn.Module,
    tokenizer: Tokenizer,
    device: torch.device,
    start_text: str,
    end_tokens: set[str] | None = None,
    max_tokens: int = 1024,
    temperature: float = 1.0,
    top_p: float = 0.9,
) -> list:
    model.eval()
    input_ids = tokenizer.encode(start_text)
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    end_ids = {tokenizer.inv_vocab[s.encode(encoding="utf-8")] for s in end_tokens if s.encode(encoding="utf-8") in tokenizer.inv_vocab} if end_tokens else None
    print(end_ids)
    with torch.no_grad():
        decoded_ids = decode(
            model=model,
            input_ids=input_ids,
            end_tokens=end_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
    cpu_ids = decoded_ids[0].detach().cpu().tolist()
    print(cpu_ids)
    return tokenizer.decode(cpu_ids)

def decode(
    model: nn.Module,
    input_ids: torch.Tensor,
    end_tokens: set[int] | None = None,
    max_tokens: int = 1024,
    temperature: float = 1.0,
    top_p: float = 0.9,
) -> torch.Tensor:
    model.eval()

    with torch.no_grad():
        while input_ids.shape[1] < max_tokens:
            logits = model(input_ids)
            logits = logits[:, -1, :]
            if temperature != 0:
                logits = logits / temperature
                probs = softmax(logits, i=-1)
            else:
                max_indices = torch.argmax(logits, dim=-1, keepdim=True)
                probs = torch.zeros_like(logits).scatter_(1, max_indices, 1.0)

            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cum_probs = torch.cumsum(sorted_probs, dim=-1)
            mask = cum_probs - sorted_probs > top_p
            sorted_probs[mask] = 0.0
            p_sorted = sorted_probs.masked_fill(mask, 0.0)
            q_sorted = p_sorted / p_sorted.sum(dim=-1, keepdim=True)
            probs = torch.zeros_like(logits).scatter_(1, sorted_indices, q_sorted)

            next_token = torch.multinomial(probs, num_samples=1)
            if end_tokens is not None and next_token.item() in end_tokens:
                break
            input_ids = torch.cat([input_ids, next_token], dim=1)

    return input_ids

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Inference Hyperparameters
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--start_text", type=str, default="Once upon a time", help="Text to start generation from")
    parser.add_argument("--special_tokens", type=int, nargs='*', default=["<|endoftext|>"], help="Tokens that indicate the end of generation")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    # Model Hyperparameters
    parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size.")
    parser.add_argument("--context_length", type=int, default=256, help="Maximum context length for the model.")
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension.")
    parser.add_argument("--d_ff", type=int, default=1344, help="Dimension of the inner feed-forward layer.")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of Transformer layers.")
    parser.add_argument("--num_heads", type=int, default=16, help="Number of attention heads.")
    parser.add_argument("--theta", type=float, default=10000.0, help="Theta value for RoPE positional embeddings.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help='Device to train on (e.g., "cpu", "cuda", "mps").',
    )
    args = parser.parse_args()
    device = args.device

    # 加载 tokenizer 
    tokenizer = Tokenizer.from_files(
        vocab_filepath="/assignment1/myoutput/TinyStoriesVocab.json",
        merges_filepath="/assignment1/myoutput/TinyStoriesMerges.txt",
        special_tokens=args.special_tokens
    )
    input_ids = tokenizer.encode(args.start_text)
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    # 加载模型
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.theta,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)

    model.to(device)
    model.eval()
    # 生成文本
    with torch.no_grad():
        output = generate(
            model=model,
            tokenizer=tokenizer,
            device=device,
            start_text=args.start_text,
            end_tokens=args.special_tokens,
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p
            )
    print("=" * 50)
    print(output)
    print("=" * 50)
