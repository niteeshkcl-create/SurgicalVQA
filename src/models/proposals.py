import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, x, mask=None):
        # x: [batch, seq_len, d_model]
        x2 = self.norm1(x)
        x = x + self.dropout1(self.self_attn(x2, x2, x2, key_padding_mask=mask)[0])
        x2 = self.norm2(x)
        x = x + self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(x2)))))
        return x

class ActionFormer(nn.Module):
    """
    Simplified ActionFormer for temporal proposal generation.
    Predicts (start, end, label, score) for each time step.
    """
    def __init__(self, input_dim=768, hidden_dim=512, num_layers=4, nhead=8, num_classes=100):
        super(ActionFormer, self).__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, nhead) for _ in range(num_layers)
        ])
        
        self.cls_head = nn.Linear(hidden_dim, num_classes)
        self.reg_head = nn.Linear(hidden_dim, 2) # [distance_to_start, distance_to_end]

    def forward(self, x, mask=None):
        """
        x: [batch, seq_len, input_dim] - Sequence of VideoMAE features
        """
        x = self.input_proj(x)
        
        for layer in self.layers:
            x = layer(x, mask=mask)
            
        cls_logits = self.cls_head(x) # [batch, seq_len, num_classes]
        reg_offsets = self.reg_head(x) # [batch, seq_len, 2]
        
        return {
            'cls_logits': cls_logits,
            'reg_offsets': reg_offsets
        }

    @torch.no_grad()
    def get_proposals(self, x, score_threshold=0.3):
        """
        Post-processing to extract proposals.
        """
        outputs = self.forward(x)
        cls_probs = torch.sigmoid(outputs['cls_logits'])
        reg_offsets = outputs['reg_offsets']
        
        batch_size, seq_len, num_classes = cls_probs.shape
        proposals = []
        
        for b in range(batch_size):
            b_proposals = []
            for t in range(seq_len):
                max_score, label = torch.max(cls_probs[b, t], dim=0)
                if max_score > score_threshold:
                    d_start, d_end = reg_offsets[b, t]
                    start = max(0, t - d_start.item())
                    end = min(seq_len - 1, t + d_end.item())
                    b_proposals.append({
                        'segment': (start, end),
                        'score': max_score.item(),
                        'label': label.item()
                    })
            # TODO: Add NMS (Non-Maximum Suppression) here
            proposals.append(b_proposals)
            
        return proposals
