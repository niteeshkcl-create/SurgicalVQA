import torch
import torch.nn as nn
from src.models.backbone import SurgicalVideoMAE
from src.models.proposals import ActionFormer
from src.models.adapter import ProposalConditionedQFormer, LLMAdapter

class SurgicalVQA(nn.Module):
    """
    SurgicalVQA: VideoMAE -> ActionFormer -> Conditioned Q-Former -> LLM
    """
    def __init__(self, 
                 num_classes_dict=None, 
                 llm_dim=4096, 
                 use_flash_attn=True):
        super(SurgicalVQA, self).__init__()
        
        # 1. Video Backbone
        self.backbone = SurgicalVideoMAE(num_classes_dict=num_classes_dict, use_flash_attn=use_flash_attn)
        
        # 2. Temporal Proposal Generator
        self.proposals_gen = ActionFormer(
            input_dim=self.backbone.config.hidden_size,
            num_classes=num_classes_dict['triplet'] if num_classes_dict else 100
        )
        
        # 3. Conditioned Q-Former
        self.q_former = ProposalConditionedQFormer(
            vision_dim=self.backbone.config.hidden_size
        )
        
        # 4. LLM Adapter
        self.llm_proj = LLMAdapter(
            qformer_dim=768, # Q-Former hidden_dim
            llm_dim=llm_dim
        )
        
        # 5. LLM (Frozen)
        # In practice, this would be a loaded HF model like LLaVA or Qwen-VL
        self.llm = None 

    def forward(self, pixel_values, question_tokens=None, labels=None):
        """
        pixel_values: [batch, C, T, H, W]
        """
        # Step 1: Extract features
        vision_features = self.backbone.extract_features(pixel_values) # [B, T_feat, D]
        
        # Step 2: Generate proposals
        # We might want to use cached proposals during training to save memory
        proposal_outputs = self.proposals_gen(vision_features)
        
        # Extract proposal metadata for conditioning [B, num_props, 3] (start, end, label)
        # Simplified: take top K proposals
        proposals_metadata = self._extract_proposal_metadata(proposal_outputs)
        
        # Step 3: Conditioned Feature Extraction
        query_output = self.q_former(vision_features, proposals=proposals_metadata)
        
        # Step 4: Map to LLM space
        llm_inputs = self.llm_proj(query_output)
        
        # Step 5: LLM forward pass (Placeholder)
        if self.llm is not None:
            # combine llm_inputs with question_tokens and generate answer
            pass
            
        return {
            'llm_inputs': llm_inputs,
            'proposals': proposal_outputs
        }

    def _extract_proposal_metadata(self, proposal_outputs, k=5):
        # Extract top K proposals based on classification scores
        cls_logits = proposal_outputs['cls_logits']
        reg_offsets = proposal_outputs['reg_offsets']
        B, T, C = cls_logits.shape
        
        scores, labels = torch.max(torch.sigmoid(cls_logits), dim=-1) # [B, T]
        top_scores, top_indices = torch.topk(scores, k, dim=-1) # [B, k]
        
        batch_indices = torch.arange(B).unsqueeze(-1).expand(-1, k)
        
        # Get start/end offsets for top indices
        top_offsets = reg_offsets[batch_indices, top_indices] # [B, k, 2]
        top_labels = labels[batch_indices, top_indices].unsqueeze(-1).float() # [B, k, 1]
        
        # Normalize time indices
        top_indices_norm = top_indices.float() / T
        start_norm = (top_indices.float() - top_offsets[:, :, 0]) / T
        end_norm = (top_indices.float() + top_offsets[:, :, 1]) / T
        
        # Proposal metadata: [start, end, label]
        metadata = torch.stack([start_norm, end_norm, top_labels.squeeze(-1)], dim=-1)
        return metadata
