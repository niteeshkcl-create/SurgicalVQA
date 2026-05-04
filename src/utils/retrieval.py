import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import faiss
import numpy as np
import os

class CLIPRetriever:
    """
    Retrieval-Augmented Temporal Context using CLIP.
    Indexes video frames and retrieves relevant context for VQA.
    """
    def __init__(self, model_name="openai/clip-vit-base-patch32", device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.index = None
        self.frame_paths = []

    @torch.no_grad()
    def build_index(self, video_root):
        """
        Scan video_root for frames and build a FAISS index.
        """
        all_embeddings = []
        self.frame_paths = []
        
        # Walk through video directories
        for root, dirs, files in os.walk(video_root):
            for file in files:
                if file.endswith(('.jpg', '.png')):
                    path = os.path.join(root, file)
                    self.frame_paths.append(path)
        
        print(f"Indexing {len(self.frame_paths)} frames...")
        
        # Batch processing
        batch_size = 32
        for i in range(0, len(self.frame_paths), batch_size):
            batch_paths = self.frame_paths[i:i+batch_size]
            images = [Image.open(p) for p in batch_paths]
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            
            image_features = self.model.get_image_features(**inputs)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            all_embeddings.append(image_features.cpu().numpy())
            
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        
        # Build FAISS index
        d = all_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d) # Inner product for cosine similarity
        self.index.add(all_embeddings.astype('float32'))
        print("Index built successfully.")

    @torch.no_grad()
    def retrieve(self, query_text=None, query_image=None, k=5):
        """
        Retrieve K most similar frames based on text or image query.
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")
            
        if query_text:
            inputs = self.processor(text=[query_text], return_tensors="pt").to(self.device)
            query_features = self.model.get_text_features(**inputs)
        elif query_image:
            inputs = self.processor(images=[query_image], return_tensors="pt").to(self.device)
            query_features = self.model.get_image_features(**inputs)
        else:
            raise ValueError("Must provide either query_text or query_image")
            
        query_features /= query_features.norm(dim=-1, keepdim=True)
        query_features = query_features.cpu().numpy().astype('float32')
        
        distances, indices = self.index.search(query_features, k)
        
        results = []
        for idx in indices[0]:
            results.append(self.frame_paths[idx])
            
        return results

    def save_index(self, path):
        faiss.write_index(self.index, f"{path}.index")
        np.save(f"{path}_paths.npy", np.array(self.frame_paths))

    def load_index(self, path):
        self.index = faiss.read_index(f"{path}.index")
        self.frame_paths = np.load(f"{path}_paths.npy").tolist()
