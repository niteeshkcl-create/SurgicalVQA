# SurgicalVQA

Video Question Answering for Surgical Phase Recognition with Temporal Grounding, powered by surgical action triplet recognition on **CholecT50**.

## Quick Start

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Dataset Preparation**:
    Download the [CholecT50](https://github.com/CAMMA-public/cholect50) dataset and place it in `data/raw/cholect50/`. The structure should be:
    
    ```
    data/raw/cholect50/
    ├── videos/
    │   ├── VID01/ (png frames)
    │   └── ...
    └── labels/
        ├── VID01.json
        └── ...
    ```

3.  **Train Baseline (Triplets + Phases)**:
    ```bash
    python scripts/train_baseline.py --data_root data/raw/cholect50 --use_flash_attn
    ```

## Project Structure

- `src/models/`: Model architectures (VideoMAE, ActionFormer, Q-Former).
- `src/data/`: Dataset loaders and transforms.
- `scripts/`: Training and utility scripts.
- `benchmarks/`: Performance benchmarks.

## Roadmap

- [x] Phase 1: Foundation & Baseline
- [ ] Phase 2: Temporal Proposals
- [ ] Phase 3: Multimodal VQA Head
- [ ] Phase 4: Retrieval & Efficiency
