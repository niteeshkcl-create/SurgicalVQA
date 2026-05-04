import torch
import torch.nn as nn
from transformers import VideoMAEForVideoClassification, VideoMAEConfig

class SurgicalVideoMAE(nn.Module):
    def __init__(self, num_classes_dict=None, model_name="MCG-NJU/videomae-base-finetuned-kinetics", use_flash_attn=True):
        super(SurgicalVideoMAE, self).__init__()
        
        if num_classes_dict is None:
            num_classes_dict = {'triplet': 100, 'phase': 100}
            
        # Load pre-trained VideoMAE encoder
        self.config = VideoMAEConfig.from_pretrained(model_name)
        
        attn_implementation = "flash_attention_2" if use_flash_attn else "eager"
        
        try:
            self.model = VideoMAEForVideoClassification.from_pretrained(
                model_name,
                num_labels=num_classes_dict['triplet'], # Default to triplet for main head
                ignore_mismatched_sizes=True,
                attn_implementation=attn_implementation
            )
        except Exception as e:
            print(f"Warning: Could not initialize with {attn_implementation}. Falling back to eager attention. Error: {e}")
            self.model = VideoMAEForVideoClassification.from_pretrained(
                model_name,
                num_labels=num_classes_dict['triplet'],
                ignore_mismatched_sizes=True
            )

        # Add additional heads for multi-task learning
        hidden_size = self.config.hidden_size
        self.heads = nn.ModuleDict({
            name: nn.Linear(hidden_size, num_classes) 
            for name, num_classes in num_classes_dict.items() if name != 'triplet'
        })
        # Override the default classifier with our triplet head
        self.model.classifier = nn.Linear(hidden_size, num_classes_dict['triplet'])

    def forward(self, pixel_values, labels=None):
        """
        pixel_values: [batch_size, num_channels, num_frames, height, width]
        labels: dict of labels for each task
        """
        outputs = self.model.videomae(pixel_values=pixel_values)
        sequence_output = outputs.last_hidden_state
        
        # Average pooling over sequence length for classification
        pooled_output = sequence_output.mean(dim=1)
        
        logits = {'triplet': self.model.classifier(pooled_output)}
        for name, head in self.heads.items():
            logits[name] = head(pooled_output)
            
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = 0
            for name, logit in logits.items():
                if name in labels:
                    loss += loss_fct(logit, labels[name])
        
        return {
            'loss': loss,
            'logits': logits
        }

    def extract_features(self, pixel_values):
        outputs = self.model.videomae(pixel_values=pixel_values)
        return outputs.last_hidden_state
