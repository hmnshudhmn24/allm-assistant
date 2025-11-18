# AllM-Assistant

**AllM-Assistant** is a lightweight instruction-tuned LLM focused on fitness and lifestyle guidance.
This repository contains training scripts, inference wrappers, a Gradio demo for Hugging Face Spaces,
and a sample instruction-response dataset to fine-tune a causal LM (GPT-2 by default).

## Contents
- `src/` — model/inference/training utilities
- `data/` — sample `train.jsonl` and `val.jsonl`
- `hf_space/` — Gradio demo app
- `requirements.txt` — exact package versions to reproduce the environment
- `README.md`, `model_card.md`, `LICENSE`, `.gitignore`

## Quick start (local)
1. Create and activate a virtual env:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Train (small quick demo):
   ```bash
   python src/trainer.py --model_name_or_path gpt2 --train_file data/train.jsonl --validation_file data/val.jsonl --output_dir outputs/allm --num_train_epochs 1 --per_device_train_batch_size 1
   ```
3. Inference:
   ```bash
   python src/inference.py --model_dir outputs/allm --prompt "Create a 10-minute beginner home workout for fat loss."
   ```
4. Run the demo locally:
   ```bash
   python hf_space/app.py
   ```

## Notes
- This project uses GPT-2 by default for speed. After testing, you can replace the base model with larger OSS LLMs.
- For efficient fine-tuning on limited hardware, consider using PEFT/LoRA (PEFT is included in requirements).
- The dataset included is synthetic sample data for demo and testing only — expand with high-quality real data for production.
