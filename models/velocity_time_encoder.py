import torch
import torch.nn as nn

from models.modules.attention_layers import AttentionBlock
from models.modules.positional_embedding import SinusoidalPositionEmbeddings


class VelocityTimeEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: tuple[int],
        embedding_dim: int,
        output_embedding_dim: int,
        num_attn_blocks: int = 8,
        num_attn_heads: int = 4,
        attn_ffn_expansion: int = 2,
        dropout_rate: float = 0.3,
    ):
        super().__init__()

        # unpacking tuple of num embeddings
        velocity_num_embeddings, dstart_num_embeddings, duration_num_embeddings = num_embeddings

        self.velocity_embedding = nn.Embedding(
            num_embeddings=velocity_num_embeddings,
            embedding_dim=embedding_dim,
        )
        self.dstart_embedding = nn.Embedding(
            num_embeddings=dstart_num_embeddings,
            embedding_dim=embedding_dim,
        )
        self.duration_embedding = nn.Embedding(
            num_embeddings=duration_num_embeddings,
            embedding_dim=embedding_dim,
        )

        # projection for concatenated embeddings to embedding_dim
        self.embedding_proj = nn.Linear(3 * embedding_dim, embedding_dim)

        self.positional_embedding = SinusoidalPositionEmbeddings(embedding_dim)

        self.attn_blocks = nn.Sequential(
            *[
                AttentionBlock(embedding_dim, heads=num_attn_heads, ffn_expansion=attn_ffn_expansion, dropout_rate=dropout_rate)
                for _ in range(num_attn_blocks)
            ]
        )

        self.output_embedding = nn.Sequential(nn.SiLU(), nn.Linear(embedding_dim, output_embedding_dim))

    def forward(self, velocity: torch.Tensor, dstart: torch.Tensor, duration: torch.Tensor) -> torch.Tensor:
        # embedding, shapes: [batch_size, seq_len] -> [batch_size, seq_len, embedding_dim]
        velocity = self.velocity_embedding(velocity)
        dstart = self.dstart_embedding(dstart)
        duration = self.duration_embedding(duration)

        # concatenate embeddings
        x = torch.cat([velocity, dstart, duration], dim=-1)
        x = self.embedding_proj(x)

        # positional embedding
        positions = torch.arange(velocity.shape[1], device=velocity.device, dtype=torch.float32)
        # shape: [batch_size, seq_len, embedding_dim]
        pe = self.positional_embedding(positions)[None, :, :]

        # combining embeddings
        x = x + pe

        # attention
        x = self.attn_blocks(x)

        # aggregate outputs to single embedding
        # shapes: [batch_size, seq_len, embedding_dim] -> batch_size, embedding_dim]
        x = x.mean(dim=1)

        # output embedding
        x = self.output_embedding(x)

        return x
