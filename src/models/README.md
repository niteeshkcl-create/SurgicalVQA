# Models

This directory contains the core model architectures for SurgicalVQA.

- `backbone.py`: VideoMAE backbone with multi-task heads for action triplet and phase recognition.
- `proposals.py`: ActionFormer-based temporal proposal generator.
- `adapter.py`: Q-Former adapter for multimodal VQA grounding (Phase 3).
- `vqa_model.py`: The integrated SurgicalVQA model (Phase 3).
