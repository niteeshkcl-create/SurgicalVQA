import torch
import torch.nn as nn
import time
from transformers import VideoMAEConfig, VideoMAEModel

def benchmark_attention(num_frames=16, height=224, width=224, batch_size=1, use_flash=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = VideoMAEConfig(
        num_frames=num_frames,
        image_size=height,
        # Other defaults
    )
    
    attn_implementation = "flash_attention_2" if use_flash else "eager"
    
    print(f"Benchmarking {attn_implementation} with {num_frames} frames...")
    
    try:
        model = VideoMAEModel(config).to(device)
        # Force the attention implementation if possible (transformers >= 4.36)
        # Note: FlashAttention 2 requires specific hardware
    except Exception as e:
        print(f"Error initializing model: {e}")
        return
        
    pixel_values = torch.randn(batch_size, 3, num_frames, height, width).to(device)
    
    # Warmup
    for _ in range(5):
        _ = model(pixel_values)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    num_runs = 20
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(pixel_values)
            
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    
    # Memory profiling
    if torch.cuda.is_available():
        max_memory = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"Avg Time: {avg_time:.4f}s, Max Memory: {max_memory:.2f}MB")
    else:
        print(f"Avg Time: {avg_time:.4f}s (CPU)")

if __name__ == "__main__":
    # Test different temporal lengths
    for t in [16, 32, 64]:
        benchmark_attention(num_frames=t, use_flash=False)
        benchmark_attention(num_frames=t, use_flash=True)
        print("-" * 30)
