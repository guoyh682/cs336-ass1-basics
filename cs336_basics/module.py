import torch
import torch.nn as nn
import torch.nn.init as init
from einops import rearrange, einsum

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None , dtype: torch.dtype | None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        std = (2/(out_features + in_features)) ** 0.5
        init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T
        # return einsum(x, self.weight, "batch sequence d_in, d_out d_in -> batch sequence d_out")

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None , dtype: torch.dtype | None = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        init.trunc_normal_(self.weight, mean=0.0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
    
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None , dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype  # 保存原始数据类型
        x = x.to(torch.float32)  # 转换为float32防止溢出
        result = x * self.weight / (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt()
        # 将结果转换回原始数据类型
        return result.to(in_dtype)
    
def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None , dtype: torch.dtype | None = None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))
    
class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        theta_inv_k = theta ** (-2 * torch.arange(d_k // 2, device=device) / d_k)
        theta_list = einsum(theta_inv_k, torch.arange(max_seq_len, device=device), "half_d_k, seq -> seq half_d_k")

        self.register_buffer("cos", torch.cos(theta_list), persistent=False)
        self.register_buffer("sin", torch.sin(theta_list), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # cos, sin: (T, D/2)
        cos = self.cos[token_positions] # (B, T, D/2)
        sin = self.sin[token_positions] # (B, T, D/2)
        x_pair = rearrange(x, "... seq_len (d_pair two) -> ... seq_len d_pair two", two = 2)  # (B, H, T, D/2, 2)
        rot_mat = torch.stack(
            (
                torch.stack((cos, -sin), dim = -1),
                torch.stack((sin, cos), dim = -1),
            ),
            dim = -2,
        ) # (B, T, D/2, 2, 2)

        x_rot = einsum(rot_mat, x_pair, "... t d_pair i j, ... h t d_pair j -> ... h t d_pair i") # (B, H, T, D/2, 2)
        out = rearrange(x_rot, "... seq_len d_pair two -> ... seq_len (d_pair two)", two = 2)
        return out

def softmax(x: torch.Tensor, i: int = -1) -> torch.Tensor:
    x = x - x.max(dim=i, keepdim=True).values
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=i, keepdim=True)

def scaled_dot_product_attention(
    Q: torch.Tensor, 
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
):
    QK = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / (Q.shape[-1] ** 0.5)
    if mask is not None:
        mask = mask.to(QK.device)
        QK = QK.masked_fill(~mask, float("-inf"))
    return einsum(softmax(QK, i=-1), V, "... queries keys, ... keys d_v -> ... queries d_v")

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, use_rope: bool = False, theta: float = 10000.0, max_seq_len: int = 512):
        super().__init__()
        d_k, d_v = d_model // num_heads, d_model // num_heads
        self.num_heads = num_heads
        self.q_proj = Linear(d_model, d_k * num_heads)
        self.k_proj = Linear(d_model, d_k * num_heads)
        self.v_proj = Linear(d_model, d_v * num_heads)
        self.output_proj = Linear(d_model, d_v * num_heads)
        self.use_rope = use_rope
        if use_rope:
            self.rope = RoPE(theta=theta, d_k=d_k, max_seq_len=max_seq_len)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = rearrange(Q, "... seq_len (head d_k) -> ... head seq_len d_k", head=self.num_heads)
        K = rearrange(K, "... seq_len (head d_k) -> ... head seq_len d_k", head=self.num_heads)
        V = rearrange(V, "... seq_len (head d_v) -> ... head seq_len d_v", head=self.num_heads)
        if self.use_rope:
            if token_positions is None:
                batch_shape = x.shape[:-2]
                seq_len = x.size(-2)
                token_positions = torch.arange(seq_len, device=x.device)
                token_positions = token_positions.expand(*batch_shape, seq_len)

            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        mask = torch.tril(torch.ones(x.shape[-2],x.shape[-2]), diagonal=0).bool()
        qkv = scaled_dot_product_attention(Q, K, V, mask=mask)
        qkv = rearrange(qkv, "... head seq_len d_v -> ... seq_len (head d_v)")
        return self.output_proj(qkv)

class TransformerBlock(nn.Module):
    def __init__(self, d_model:int, num_heads:int, d_ff:int, use_rope:bool = False, theta:float = 10000.0, max_seq_len:int = 512):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, use_rope=use_rope, theta=theta, max_seq_len=max_seq_len)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        y = x + self.attn(self.ln1(x), token_positions)
        return y + self.ffn(self.ln2(y))

class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, use_rope=True, theta=rope_theta, max_seq_len=context_length) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        return self.lm_head(x)

