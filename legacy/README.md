# Legacy Scripts

This folder contains the original Google Colab notebooks used during development. They are preserved for reference only.

**Use the pipeline scripts in `scripts/` instead.**

## Script Mapping

| Legacy Script | Pipeline Script |
|---|---|
| `classification/gpt_classification.py` | `scripts/run_classification.py` |
| `classification/gpt_vision_classification.py` | `scripts/run_classification.py` |
| `classification/gpt_text_vision_classification.py` | `scripts/run_classification.py` |
| `classification/gpt_v_&_tv_classification.py` | `scripts/run_classification.py` |
| `classification/gpt_v_&_tv_classification_ablation_study.py` | `scripts/run_classification.py --ablation` |
| `classification/claude_text_classification.py` | `scripts/run_classification.py` |
| `classification/claude_vision_classification.py` | `scripts/run_classification.py` |
| `classification/claude_text_&_vision_classification.py` | `scripts/run_classification.py` |
| `classification/claude_v_&_vt_classification.py` | `scripts/run_classification.py` |
| `classification/llama_classification.py` | `scripts/run_llama_classification.py` |
| `classification/copy_of_llama_classification.py` | `scripts/run_llama_classification.py` |
| `classification/parsing_classification_responses.py` | `scripts/compute_metrics.py` |
| `generation/gpt_generation.py` | `scripts/run_generation.py` |
| `representation/fine_tuning_clip_w_batch_sizes,_learning_rates,_and_usage.py` | `scripts/run_clip_finetuning.py` |
| `representation/fine_tuning_clip_using_adafactor_automatic_lr_w_batch_sizes,_learning_rates,_and_usage.py` | `scripts/run_clip_finetuning.py` (older variant) |
| `representation/alignment_&_uniformity_fine_tuning_clip_using_triplet_loss.py` | (analysis-only, not ported) |
| `representation/clip_similarity_testing.py` | `scripts/run_clip_similarity.py` |
