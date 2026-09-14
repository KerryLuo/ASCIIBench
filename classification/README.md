# Classification (Original Scripts)

These are the original single-model, single-modality classification scripts used during development. They were written for Google Colab and reference Google Drive paths.

**Do not use these directly.** Use the unified pipeline scripts in the repo root instead:

- `run_classification.py` — OpenAI and Anthropic models (text, vision, text+vision)
- `run_llama_classification.py` — LLaMA models (local GPU)
- `compute_metrics.py` — Metrics computation from results

The scripts here are kept for reference only.

### Script mapping

| Original script | Superseded by | Notes |
|---|---|---|
| `gpt_v_&_tv_classification.py` | `run_classification.py` | Most modern GPT vision/text+vision script |
| `gpt_v_&_tv_classification_ablation_study.py` | `run_classification.py --ablation` | Inverted-colors variant |
| `gpt_classification.py` | `run_classification.py` | Text-only GPT |
| `gpt_vision_classification.py` | `run_classification.py` | Older vision script (outdated preprocessing) |
| `gpt_text_vision_classification.py` | `run_classification.py` | Older text+vision script (outdated preprocessing) |
| `claude_v_&_vt_classification.py` | `run_classification.py` | Most modern Claude vision/text+vision script |
| `claude_text_classification.py` | `run_classification.py` | Text-only Claude |
| `claude_vision_classification.py` | `run_classification.py` | Older Claude vision script |
| `claude_text_&_vision_classification.py` | `run_classification.py` | Older Claude text+vision script (outdated preprocessing) |
| `llama_classification.py` | `run_llama_classification.py` | Most modern LLaMA script |
| `copy_of_llama_classification.py` | `run_llama_classification.py` | Older copy, missing resume logic |
| `parsing_classification_responses.py` | `compute_metrics.py` | Post-processing and metrics utilities |
