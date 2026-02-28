"""Post-training: aligning a base model for useful, safe behavior.

Sub-stages:
- SFT: Supervised fine-tuning on (prompt, response) pairs
- RL: Reinforcement learning from human feedback (RLHF, DPO, etc.) [not yet implemented]
"""
