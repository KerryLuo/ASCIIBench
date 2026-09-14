# ASCIIBench: Evaluating Language-Model-Based Understanding of Visually-Oriented Text

A benchmark for evaluating how well large language models classify ASCII art. The dataset contains **5,315 items across 752 classes**. Models are evaluated using a 4-choice classification task across three modalities: text-only, vision-only, and text+vision.

Paper: [arxiv.org/abs/2512.04125](https://arxiv.org/abs/2512.04125)

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/KerryLuo/ASCIIBench.git
cd ASCIIBench

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up API keys
cp secrets/.env.example secrets/.env
# Edit secrets/.env and add your API keys

# 4. Run classification
python run_classification.py
```

## Setup

### Requirements

- Python 3.8+
- API keys for [OpenAI](https://platform.openai.com/) and/or [Anthropic](https://console.anthropic.com/)
- For LLaMA: CUDA GPU + `pip install -r requirements-llama.txt`

### API Keys

Create `secrets/.env` from the template:

```bash
cp secrets/.env.example secrets/.env
```

Then add your keys:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

Keys are loaded automatically — never commit `secrets/.env` to version control.

## Usage

### API Models (OpenAI & Anthropic)

```bash
# Run all models and modalities from config.yaml
python run_classification.py

# Run specific model(s)
python run_classification.py --models gpt-4o claude-3-5-sonnet-20240620

# Run specific modalities
python run_classification.py --modalities text vision

# Run ablation study (inverted colors: black background, white text)
python run_classification.py --ablation

# Custom dataset path
python run_classification.py --dataset path/to/dataset.jsonl
```

Runs save incrementally — if interrupted, re-run the same command and it resumes where it left off.

### LLaMA (Local GPU)

```bash
pip install -r requirements-llama.txt

# Run both LLaMA models
python run_llama_classification.py

# Run a specific model
python run_llama_classification.py --models meta-llama/Meta-Llama-3.1-8B-Instruct
```

### Compute Metrics

```bash
# Compute metrics for all results
python compute_metrics.py

# Specific result file(s)
python compute_metrics.py results/gpt-4o_text_results.jsonl

# Remove duplicates and exclude parse errors
python compute_metrics.py --dedupe --filter-errors

# Per-class breakdown
python compute_metrics.py --per-class

# CSV output
python compute_metrics.py --csv
```

Metrics reported: **micro accuracy**, **macro accuracy** (mean of per-class accuracies), and **pass rate** (fraction of responses that parsed successfully).

## Configuration

Edit `config.yaml` to customize models, modalities, and preprocessing:

```yaml
models:
  - name: gpt-4o
    provider: openai
    modalities: [text, vision, text_vision]

  - name: claude-3-5-sonnet-20240620
    provider: anthropic
    modalities: [text, vision, text_vision]

preprocessing:
  font_size: 18
  image_size: [1600, 1200]
  background: white      # "black" for ablation
  text_color: black       # "white" for ablation
```

CLI flags (`--models`, `--modalities`, `--ablation`) override config values.

## Project Structure

```
ASCIIBench/
├── final_dataset.jsonl          # Dataset (5,315 ASCII art items)
├── config.yaml                  # Pipeline configuration
├── run_classification.py        # Main pipeline (OpenAI + Anthropic)
├── run_llama_classification.py  # LLaMA pipeline (local GPU)
├── compute_metrics.py           # Metrics computation
├── requirements.txt             # Python dependencies
├── requirements-llama.txt       # Additional deps for LLaMA
├── secrets/
│   └── .env.example             # API key template
├── results/                     # Classification results (JSONL)
└── Classification/              # Original evaluation scripts
```

## Results Format

Each result file is newline-delimited JSON (`.jsonl`). Each line:

```json
{
  "model": "gpt-4o",
  "modality": "text",
  "ascii_art": "...",
  "choices": ["omega", "bird", "ocean", "land"],
  "predicted_class": "omega",
  "actual_class": "omega",
  "correct": true,
  "response": "..."
}
```

## Models Evaluated

| Model | Provider | Modalities |
|-------|----------|------------|
| GPT-4o | OpenAI | text, vision, text+vision |
| GPT-4o-mini | OpenAI | text, vision, text+vision |
| GPT-5-mini | OpenAI | text, vision, text+vision |
| GPT-3.5-turbo | OpenAI | text |
| Claude 3.5 Sonnet | Anthropic | text, vision, text+vision |
| LLaMA 3.1-8B | Local | text |
| LLaMA 3.1-8B-Instruct | Local | text |

## Citation

If you use this code or dataset, please cite:

```bib
@misc{luo2025asciibenchevaluatinglanguagemodelbasedunderstanding,
      title={ASCIIBench: Evaluating Language-Model-Based Understanding of Visually-Oriented Text}, 
      author={Kerry Luo and Michael Fu and Joshua Peguero and Husnain Malik and Anvay Patil and Joyce Lin and Megan Van Overborg and Ryan Sarmiento and Kevin Zhu},
      year={2025},
      eprint={2512.04125},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2512.04125}, 
}
```
