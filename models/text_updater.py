import torch
import torch.nn as nn
from diffusers.models import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention


class RelationAttentionWithSelfAttention(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
            self, 
            query_dim, 
            sg_emb_dim,
            n_heads,
            d_head,
        ):
        super().__init__()

        self.linear = nn.Sequential(
            nn.Linear(sg_emb_dim, query_dim),
            nn.SiLU(),
        ) 
        self.attn = Attention(query_dim=query_dim, heads=n_heads, dim_head=d_head)
        self.ff = FeedForward(query_dim, activation_fn="geglu")

        self.self_attn = Attention(query_dim=query_dim, heads=n_heads, dim_head=d_head)

        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)

        self.register_parameter('alpha_attn', nn.Parameter(torch.tensor(0.)))
        self.register_parameter('alpha_dense', nn.Parameter(torch.tensor(0.)))
        self.register_parameter('alpha_self', nn.Parameter(torch.tensor(0.)))

    def forward(self, x, sg_embed, cross_attention_mask=None, self_attention_mask=None):
        
        sg_embed = self.linear(sg_embed)
        # explicitly divide local token and global token
        x = x + self.alpha_attn.tanh() * self.attn(self.norm1(x), encoder_hidden_states=self.norm1(sg_embed), attention_mask=cross_attention_mask)
        x = x + self.alpha_dense.tanh() * self.ff(self.norm2(x))
        x = x + self.alpha_self.tanh() * self.self_attn(x, attention_mask=self_attention_mask)

        return x

class RelationAttention(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
            self, 
            query_dim, 
            sg_emb_dim,
            n_heads,
            d_head,
            pooling=False
        ):
        super().__init__()

        self.linear = nn.Sequential(
            nn.Linear(sg_emb_dim, query_dim),
            nn.SiLU(),
            nn.Linear(query_dim, query_dim),
            nn.SiLU(),
            nn.Linear(query_dim, query_dim),
        ) 
        self.attn = Attention(query_dim=query_dim, heads=n_heads, dim_head=d_head)
        self.ff = FeedForward(query_dim, activation_fn="geglu")

        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)

        self.register_parameter('alpha_attn', nn.Parameter(torch.tensor(0.)))
        self.register_parameter('alpha_dense', nn.Parameter(torch.tensor(0.)))

        self.pooling = pooling
        if self.pooling:
            self.register_parameter('alpha_pool', nn.Parameter(torch.tensor(-5.0)))

    def forward(self, x, sg_embed, cross_attention_mask=None, self_attention_mask=None):
        
        sg_embed = self.linear(sg_embed)
        # explicitly divide local token and global token
        x = x + self.alpha_attn.tanh() * self.attn(self.norm1(x), encoder_hidden_states=self.norm1(sg_embed), attention_mask=cross_attention_mask)
        x = x + self.alpha_dense.tanh() * self.ff(self.norm2(x))

        if cross_attention_mask is not None and self.pooling:
            # Normalize the attention mask to have zeros where tokens are not involved,
            # which is already the case as per your description.
            attention_mask_inverted = cross_attention_mask.clone()
            attention_mask_inverted[cross_attention_mask == 0] = 1
            attention_mask_inverted[cross_attention_mask != 0] = 0

            # Step 4: (New Step) Relation-aware pooling operation
            # Sum token embeddings for each relation based on the normalized attention mask
            relation_embedding_sums = torch.einsum('bji,bjd->bid', attention_mask_inverted, x)

            # Calculate the number of tokens involved in each relation to average the sums correctly
            token_counts_per_relation = attention_mask_inverted.sum(dim=1, keepdim=True) + 1e-5  # Avoid division by zero
            token_counts_per_relation = token_counts_per_relation.squeeze(1)

            # Now compute the average embedding for each relation
            relation_embedding_averages = relation_embedding_sums / token_counts_per_relation.unsqueeze(-1)

            # Step 5: (New Step) Update token embeddings based on the relation averages
            # We use the normalized attention mask again to sum the relevant relation averages for each token
            updated_token_embeddings = torch.einsum('bij,bjd->bid', attention_mask_inverted, relation_embedding_averages)

            # Introduce a learnable gate to control the update strength
            update_gate = torch.sigmoid(self.alpha_pool)  # Assuming alpha_attn is used as the gate controller

            # Interpolate between the original and updated embeddings
            x = (1 - update_gate) * x + update_gate * updated_token_embeddings

        return x

