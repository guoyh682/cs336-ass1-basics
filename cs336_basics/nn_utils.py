import torch
import math
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import os
import typing
from collections.abc import Iterable
from torch.optim import Optimizer
from einops import rearrange, einsum

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits = rearrange(logits, "batch ... vocab -> (batch ...) vocab")
    targets = rearrange(targets, "batch ... -> (batch ...)")
    # 减去最大保证稳定
    stable_logits = logits - logits.max(dim=-1, keepdim=True).values
    sum_logits = torch.log(einsum(torch.exp(stable_logits), "b v -> b"))
    target_logits = stable_logits.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
    loss = torch.mean(-target_logits + sum_logits, dim=-1)
    return loss

class AdamW(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    def step(self, closure=None):
        """执行单次参数更新
        参数:
            closure: 用于重新计算损失的可调用函数(可选)
        返回:
            计算出的损失值(如果提供了closure)
        """
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # 获取当前学习率
            beta1, beta2 = group["betas"]  # 获取一阶和二阶矩估计的衰减率
            eps = group["eps"]  # 获取数值稳定性常数
            weight_decay = group["weight_decay"]  # 获取权重衰减系数

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # 获取参数p的状态字典
                t = state.get("t", 0)  # 从状态中获取迭代次数，默认为0
                m = state.get("m", 0)  # first moment estimate
                v = state.get("v", 0)
                grad = p.grad.data  # 获取损失函数关于p的梯度
                
                # 执行带学习率衰减的更新(核心公式)
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad**2
                lr_t = lr * ((1 - beta2**(t+1)) ** 0.5) / (1 - beta1**(t+1))
                p.data -= lr_t * m / (v ** 0.5 + eps)
                p.data -= lr * weight_decay * p.data

                state["t"] = t + 1  # 更新迭代计数器
                state["m"] = m
                state["v"] = v
        return loss
    
def cosine_lr_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if it < warmup_iters:
        lr = max_learning_rate * it / warmup_iters
    elif it > cosine_cycle_iters:
        lr = min_learning_rate
    else:
        lr = min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (
            1 + math.cos(math.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters))
        )
    return lr

def gradient_clipping(params: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6):
    norm_sum = 0
    grads = [p.grad.data for p in params if p.grad is not None]
    norm_sum = sum(grad.norm(2) ** 2 for grad in grads)
    if norm_sum > max_l2_norm:
        for p in params:
            if p.grad is None:
                continue
            p.grad.data = p.grad.data * (max_l2_norm / (norm_sum ** 0.5 + eps))

def load_data(
    dataset, batch_size: int, context_length: int, device: str
):
    dataset = torch.from_numpy(dataset).long().to(device)
    sample_num = dataset.shape[0]
    max_start = sample_num - context_length - 1
    starts = np.random.randint(0, max_start + 1, size=batch_size)
    inputs, labels = [], []
    for start in starts:
        end = start + context_length
        inputs.append(dataset[start:end])
        labels.append(dataset[start + 1:end + 1])
    return torch.stack(inputs,dim=0), torch.stack(labels, dim=0)

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]): 
## 应将前三个参数的所有状态转储到文件类对象out中。
## 你可以使用模型和优化器的state_dict方法获取它们相关状态，
## 并使用torch.save(obj, out)将obj转储到out中（PyTorch支持路径或文件类对象）。
## 通常选择让obj成为一个字典，但只要之后能加载你的检查点，你可以使用任何格式。
    model_dict = model.state_dict()
    opt_dict = optimizer.state_dict()
    torch.save({
        'model': model_dict,
        'optimizer': opt_dict,
        'iteration': iteration
    }, out)

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
    ## 应从src（路径或文件类对象）加载检查点，然后从该检查点恢复模型和优化器状态。
    ## 你的函数应返回保存到检查点的迭代次数。
    ## 你可以使用torch.load(src)恢复你在save_checkpoint实现中保存的内容，
    ##并使用模型和优化器中的load_state_dict方法将它们恢复到之前的状态。
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['iteration']



if __name__ == "__main__":
    # 创建可训练参数 (10x10随机矩阵，初始标准差为5)
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))

    # 初始化SGD优化器(学习率设为1)
    opt = torch.optim.SGD([weights], lr=100)

    # 训练循环
    for t in range(10):
        opt.zero_grad()                  # 重置所有可训练参数的梯度
        loss = (weights**2).mean()       # 计算标量损失值(示例使用L2正则)
        print(loss.cpu().item())         # 打印当前损失值
        loss.backward()                  # 反向传播计算梯度
        opt.step()                       # 执行优化器更新