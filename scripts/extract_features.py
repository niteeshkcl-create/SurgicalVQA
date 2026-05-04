import os
import sys
import argparse
import torch

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from src.data.dataset import CholecT50Dataset, get_transforms
from src.models.backbone import SurgicalVideoMAE

def extract_features(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load all videos (1-100+ depending on CholecT50 version)
    # We'll just iterate through what's in data_root/videos
    video_dirs = sorted([d for d in os.listdir(os.path.join(args.data_root, 'videos')) if d.startswith('VID')])
    video_ids = [int(d[3:]) for d in video_dirs]
    
    # Use evaluation transforms
    transform = get_transforms('val')
    
    model = SurgicalVideoMAE(use_flash_attn=args.use_flash_attn)
    model.to(device)
    model.eval()
    
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded checkpoint from {args.checkpoint}")

    os.makedirs(args.output_dir, exist_ok=True)

    for vid_id in video_ids:
        print(f"Extracting features for VID{vid_id:02d}...")
        dataset = CholecT50Dataset(args.data_root, [vid_id], transform=transform)
        # Use a large batch size for faster extraction
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        
        all_features = []
        
        with torch.no_grad():
            for clips, _ in tqdm(loader):
                clips = clips.to(device)
                # extract_features returns [batch, seq_len, hidden_size]
                # We pool over sequence length or keep it depending on requirements
                # ActionFormer usually expects a single vector per time step
                features = model.extract_features(clips)
                # Mean pool over the 16 frames in the clip
                features = features.mean(dim=1) # [batch, hidden_size]
                all_features.append(features.cpu().numpy())
                
        all_features = np.concatenate(all_features, axis=0)
        np.save(os.path.join(args.output_dir, f"VID{vid_id:02d}.npy"), all_features)
        print(f"Saved {all_features.shape} to VID{vid_id:02d}.npy")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="data/cache/features")
    parser.add_argument("--checkpoint", type=str, help="Path to fine-tuned VideoMAE checkpoint")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_flash_attn", action="store_true")
    
    args = parser.parse_args()
    extract_features(args)
