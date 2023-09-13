import torch
import einops
import torch.nn as nn


class ConvMultiHeadAttention(nn.Module):
    def __init__(self, channels: int, heads: int = 4):
        """
        Multi Head Attention Layer

        Args:
            channels (int): number of channels
            heads (int): number of attention heads
        """

        super().__init__()

        self.heads = heads
        self.head_dim = channels // heads
        self.embedding_size = channels

        assert self.head_dim * heads == channels, "Channels needs to be dividable by heads"

        self.values_proj = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.keys_proj = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.queries_proj = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.out = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # applying linear projections
        q = self.queries_proj(x)
        k = self.keys_proj(x)
        v = self.values_proj(x)

        # rearranging q, k, v from [batch_size, channels, seq_len] -> [batch_size, seq_len, heads, head_dim]
        q = einops.rearrange(q, "n (h d) l -> n l h d", h=self.heads)
        k = einops.rearrange(k, "n (h d) l -> n l h d", h=self.heads)
        v = einops.rearrange(v, "n (h d) l -> n l h d", h=self.heads)

        # shapes
        # query: (N, query_len, heads, head_dim)
        # keys: (N, key_len, heads, head_dim)
        # output: (N, heads, query_len, key_len)
        qk = torch.einsum("n q h d, n k h d -> n h q k", [q, k])

        # applying softmax over key dimension to calculate attention scores
        attn = torch.softmax(qk * (self.embedding_size**-0.5), dim=3)

        # shapes
        # attn: (N, heads, query_len, key_len)
        # values: (N, values_len, heads, head_dim)
        # output: (N, query_len, heads, head_dim)
        out = torch.einsum("n h q l, n l h d -> n q h d", [attn, v])

        # concatenation of heads
        out = einops.rearrange(out, "n l h d -> n (h d) l")

        return self.out(out)


class CrossAttention(nn.Module):
    def __init__(self, channels: int, label_embedding_dim: int, heads: int = 4):
        """
        Multi Head Attention Layer

        Args:
            channels (int): number of channels
            label_embedding_dim (int): embeddings size of label vector
            heads (int): number of attention heads
        """

        super().__init__()

        self.heads = heads
        self.head_dim = channels // heads
        self.embedding_size = channels

        assert self.head_dim * heads == channels, "Channels needs to be dividable by heads"

        self.queries_proj = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.keys_proj = nn.Linear(label_embedding_dim, channels, bias=False)
        self.values_proj = nn.Linear(label_embedding_dim, channels, bias=False)

        self.out = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        seq_len = q.shape[-1]

        # applying linear projections
        q = self.queries_proj(q)
        k = self.keys_proj(k)
        v = self.values_proj(v)

        # rearranging q from [batch_size, channels, seq_len] -> [batch_size, seq_len, heads, head_dim]
        q = einops.rearrange(q, "n (h d) l -> n l h d", h=self.heads)

        # rearranging k, v from [batch_size, channels] -> [batch_size, seq_len, heads, head_dim]
        k = einops.rearrange(k, "n (h d) -> n h d", h=self.heads)
        k = einops.repeat(k, "n h d -> n l h d", l=seq_len)
        v = einops.rearrange(v, "n (h d) -> n h d", h=self.heads)
        v = einops.repeat(v, "n h d -> n l h d", l=seq_len)

        # shapes
        # query: (N, query_len, heads, head_dim)
        # keys: (N, key_len, heads, head_dim)
        # output: (N, heads, query_len, key_len)
        qk = torch.einsum("n q h d, n k h d -> n h q k", [q, k])

        # applying softmax over key dimension to calculate attention scores
        attn = torch.softmax(qk * (self.embedding_size**-0.5), dim=3)

        # shapes
        # attn: (N, heads, query_len, key_len)
        # values: (N, values_len, heads, head_dim)
        # output: (N, query_len, heads, head_dim)
        out = torch.einsum("n h q l, n l h d -> n q h d", [attn, v])

        # concatenation of heads
        out = einops.rearrange(out, "n l h d -> n (h d) l")

        return self.out(out)


class LinearMultiHeadAttention(nn.Module):
    def __init__(self, embedding_size: int, heads: int):
        """
        Multi Head Attention Layer

        Args:
            embedding_size (int): size of embedding dim
            heads (int): number of attention heads
        """

        super().__init__()

        self.heads = heads
        self.head_dim = embedding_size // heads
        self.embedding_size = embedding_size

        assert self.head_dim * heads == embedding_size, "Embedding size needs to be dividable by heads"

        self.values_proj = nn.Linear(embedding_size, embedding_size, bias=False)
        self.keys_proj = nn.Linear(embedding_size, embedding_size, bias=False)
        self.queries_proj = nn.Linear(embedding_size, embedding_size, bias=False)

        self.fc_out = nn.Linear(embedding_size, embedding_size)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # applying linear projections
        q = self.queries_proj(q)
        k = self.keys_proj(k)
        v = self.values_proj(v)

        # rearranging q, k, v from [batch_size, seq_len, embedding_size] -> [batch_size, seq_len, heads, head_size]
        q = einops.rearrange(q, "n l (h d) -> n l h d", h=self.heads)
        k = einops.rearrange(k, "n l (h d) -> n l h d", h=self.heads)
        v = einops.rearrange(v, "n l (h d) -> n l h d", h=self.heads)

        # shapes
        # query: (N, query_len, heads, head_dim)
        # keys: (N, key_len, heads, head_dim)
        # output: (N, heads, query_len, key_len)
        qk = torch.einsum("n q h d, n k h d -> n h q k", [q, k])

        # applying softmax over key dimension to calculate attention scores
        attn = torch.softmax(qk * (self.embedding_size**-0.5), dim=3)

        # shapes
        # attn: (N, heads, query_len, key_len)
        # values: (N, values_len, heads, head_dim)
        # output: (N, query_len, heads, head_dim)
        out = torch.einsum("n h q l, n l h d -> n q h d", [attn, v])

        # concatenation of heads
        out = einops.rearrange(out, "n l h d -> n l (h d)")

        return self.fc_out(out)


class AttentionBlock(nn.Module):
    def __init__(self, embedding_size: int, heads: int, ffn_expansion: int, dropout_rate: float = 0.3):
        """
        Attention Block

        Args:
            embedding_size (int): size of embedding dim
            heads (int): number of attention heads
            ffn_expansion (int): scaling factor for hidden dim expansion in feed forward layer
            dropout_rate (float, optional): dropout rate. Defaults to 0.3.
        """
        super().__init__()

        # expanded dimension for feed forward
        hidden_dim = embedding_size * ffn_expansion

        self.ln1 = nn.LayerNorm(embedding_size)
        self.attention = LinearMultiHeadAttention(embedding_size, heads)

        self.ln2 = nn.LayerNorm(embedding_size)
        self.feed_forward = nn.Sequential(nn.Linear(embedding_size, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, embedding_size))

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # skip connection
        skip = x

        # normalization
        x = self.ln1(x)

        # calculating attention
        x = self.attention(x, x, x)

        # residual connection with input x
        x = x + skip

        # skip connection for feed forward
        skip = x

        # dropout
        x = self.dropout(x)

        # normalization
        x = self.ln2(x)

        # passing to feedforward layer
        x = self.feed_forward(x)

        # residual connection
        x = x + skip

        # dropout
        x = self.dropout(x)

        return x
