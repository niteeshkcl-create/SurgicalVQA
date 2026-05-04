import torch
import torch.nn as nn
from transformers import BertConfig, BertModel

class ProposalConditionedQFormer(nn.Module):
    """
    Q-Former adapter conditioned on temporal proposals.
    Extracts visual features from VideoMAE conditioned on predicted segments.
    """
    def __init__(self, vision_dim=768, cross_attention_freq=2, num_query_token=32, hidden_dim=768):
        super(ProposalConditionedQFormer, self).__init__()
        
        # We use a BERT-like architecture for the Q-Former
        self.config = BertConfig.from_pretrained("bert-base-uncased")
        self.config.query_length = num_query_token
        self.config.vision_layers = cross_attention_freq # Freq of cross-attention
        
        # Learnable query tokens
        self.query_tokens = nn.Parameter(torch.zeros(1, num_query_token, hidden_dim))
        self.query_tokens.data.normal_(mean=0.0, std=self.config.initializer_range)
        
        # The Q-Former itself (BERT with cross-attention)
        # In a real implementation, we would modify BERT to include cross-attention
        # For this project, we'll use a simplified Transformer with cross-attention layers
        self.qformer = BertModel(self.config, add_pooling_layer=False)
        
        # Projection from vision_dim to Q-Former hidden_dim
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        
        # Temporal Proposal Conditioning: 
        # Embeddings for (start, end, label)
        self.segment_embedding = nn.Linear(3, hidden_dim) # [start_norm, end_norm, label_id]

    def forward(self, vision_features, proposals=None):
        """
        vision_features: [batch, seq_len, vision_dim]
        proposals: List of proposal metadata or a tensor of [batch, max_proposals, 3]
        """
        batch_size = vision_features.shape[0]
        
        # Project vision features
        vision_features = self.vision_proj(vision_features)
        
        # Initialize queries
        queries = self.query_tokens.expand(batch_size, -1, -1)
        
        # Inject proposal conditioning if provided
        if proposals is not None:
            # Simple conditioning: Add average segment embedding to queries
            # In a more advanced version, we would have per-proposal queries
            seg_emb = self.segment_embedding(proposals) # [batch, num_proposals, hidden_dim]
            # Pool proposals or select top K
            seg_emb = seg_emb.mean(dim=1, keepdim=True)
            queries = queries + seg_emb
            
        # Q-Former cross-attention (simplified view)
        # In BERT, we pass vision_features as encoder_hidden_states
        outputs = self.qformer(
            inputs_embeds=queries,
            encoder_hidden_states=vision_features,
            return_dict=True
        )
        
        return outputs.last_hidden_state # [batch, num_query_token, hidden_dim]

class LLMAdapter(nn.Module):
    """
    Connects Q-Former output to LLM input space.
    """
    def __init__(self, qformer_dim=768, llm_dim=4096):
        super(LLMAdapter, self).__init__()
        self.proj = nn.Linear(qformer_dim, llm_dim)

    def forward(self, x):
        return self.proj(x)
