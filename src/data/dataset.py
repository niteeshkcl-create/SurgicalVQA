import os
import json
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from torchvision import transforms

class CholecT50Dataset(Dataset):
    """
    CholecT50 dataset for surgical action triplet and phase recognition.
    Triplets: <instrument, verb, target>
    """
    def __init__(self, data_root, video_list, num_frames=16, frame_stride=5, transform=None, mode='train'):
        self.data_root = data_root
        self.video_list = [f"VID{str(v).zfill(2)}" for v in video_list]
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.transform = transform
        self.mode = mode
        
        self.samples = self._prepare_samples()

    def _prepare_samples(self):
        samples = []
        for video_id in self.video_list:
            label_file = os.path.join(self.data_root, 'labels', f'{video_id}.json')
            if not os.path.exists(label_file):
                continue
                
            with open(label_file, 'r') as f:
                data = json.load(f)
                annotations = data['annotations']
                
                for frame_id, labels in annotations.items():
                    # labels is a list of [triplet, tool, verb, target, phase]
                    # We convert these to binary vectors for multi-label classification
                    triplet_label, tool_label, verb_label, target_label, phase_label = self._get_binary_labels(labels)
                    
                    samples.append({
                        'video_id': video_id,
                        'frame_id': int(frame_id),
                        'triplet_label': triplet_label,
                        'tool_label': tool_label,
                        'verb_label': verb_label,
                        'target_label': target_label,
                        'phase_label': phase_label
                    })
        return samples

    def _get_binary_labels(self, labels):
        # CholecT50 counts: 100 triplets, 6 tools, 10 verbs, 15 targets, 100 phases (per repo code)
        # Note: CholecT50 phase labels are often more granular than Cholec80
        triplet_label = np.zeros(100)
        tool_label = np.zeros(6)
        verb_label = np.zeros(10)
        target_label = np.zeros(15)
        phase_label = np.zeros(100) # Adjust based on actual dataset usage

        for label in labels:
            if label[0] != -1: triplet_label[int(label[0])] = 1
            # label[1:7] are binary tools in some versions, but here we expect single tool indices
            # Let's follow the repo's logic for tool extraction if possible
            # The repo says label[1:7] is tool... wait, let me re-check that.
            # Repo: tool = label[1:7]; if tool[0] != -1.0: tool_label[tool[0]] += 1
            # This suggests label[1] is an index or start of binary?
            # Re-reading: label[1:7] is tool. If label[1] is an index...
            # Actually, CholecT50 triplets are usually single per action.
            if label[1] != -1: tool_label[int(label[1])] = 1
            if label[7] != -1: verb_label[int(label[7])] = 1
            if label[8] != -1: target_label[int(label[8])] = 1
            if label[14] != -1: phase_label[int(label[14])] = 1
            
        return triplet_label, tool_label, verb_label, target_label, phase_label

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_id = sample['video_id']
        frame_id = sample['frame_id']
        
        frames = []
        for i in range(self.num_frames):
            offset = (i - self.num_frames // 2) * self.frame_stride
            curr_frame_id = frame_id + offset
            
            # CholecT50 frames are usually stored as VIDXX/000XXX.png
            frame_path = os.path.join(self.data_root, 'videos', video_id, f'{str(curr_frame_id).zfill(6)}.png')
            
            if os.path.exists(frame_path):
                img = Image.open(frame_path).convert('RGB')
            else:
                img = Image.new('RGB', (448, 256))
            
            if self.transform:
                img = self.transform(img)
            frames.append(img)
            
        clip = torch.stack(frames).permute(1, 0, 2, 3) # [C, T, H, W]
        
        labels = {
            'triplet': torch.tensor(sample['triplet_label'], dtype=torch.float32),
            'tool': torch.tensor(sample['tool_label'], dtype=torch.float32),
            'verb': torch.tensor(sample['verb_label'], dtype=torch.float32),
            'target': torch.tensor(sample['target_label'], dtype=torch.float32),
            'phase': torch.tensor(sample['phase_label'], dtype=torch.float32)
        }
        
        return clip, labels

def get_transforms(mode='train'):
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((256, 448)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((256, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
