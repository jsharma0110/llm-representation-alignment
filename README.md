# Representation Alignment and Model Stitching Across Modalities

A research toolkit for studying how internal representations can be compared, aligned, and transferred across neural models.

The project investigates whether representations learned by different models exhibit common structure, where representations diverge across model scales and architectures, and whether lightweight alignment methods can enable representation transfer and model stitching.

## Research Questions

This project explores:

- whether different neural models learn similar internal representations,
- how representation similarity changes across network depth,
- where smaller and larger models begin to diverge,
- whether representation differences relate to missing or incorrectly expressed knowledge,
- whether learned mappings between representation spaces generalize to held-out data,
- and whether aligned representations can support model stitching across architectures and modalities.

## Current Work

### LLM Representation Alignment

Current language-model comparisons include:

- TinyLlama-1.1B-Chat vs. Qwen2.5-0.5B-Instruct
- Llama-3.2-1B-Instruct vs. Llama-3.2-3B-Instruct

Implemented analyses include:

- hidden-state extraction,
- token-level representation pooling,
- layer-wise Linear CKA,
- PCA-based dimensionality reduction,
- orthogonal Procrustes similarity,
- held-out Procrustes alignment,
- shuffled-prompt controls,
- and layer-wise similarity visualization.

Additional work investigates direct matching and representation-based model stitching.

### Cross-Modal Alignment

The broader project also studies representation alignment across speech, audio-language models, and physiological time-series data, where differences in temporal resolution introduce additional alignment challenges.

## Repository Structure

```text
llm-representation-alignment/
├── data/
│   └── factual_qa/
│
├── src/
│   └── alignment/
│       ├── datasets/
│       ├── extraction/
│       ├── models/
│       ├── similarity/
│       │   ├── cka.py
│       │   └── procrustes.py
│       ├── visualization/
│       ├── benchmarking/
│       └── utils/
│
├── experiments/
│   ├── llm/
│   │   ├── tinyllama_qwen/
│   │   └── llama1b_llama3b/
│   ├── palaash/
│   └── ismail/
│
├── examples/
│
├── results/
│   ├── hidden_states/
│   │   ├── tinyllama/
│   │   ├── qwen/
│   │   ├── llama1b/
│   │   └── llama3b/
│   ├── tinyllama_qwen/
│   └── llama1b_llama3b/
│
├── figures/
│   ├── tinyllama_qwen/
│   └── llama1b_llama3b/
│
├── pyproject.toml
├── requirements.txt
├── README.md
└── .gitignore
```

Reusable implementations live under `src/alignment/`. Experiment directories contain lightweight runners that configure and execute those components for particular model comparisons.

Generated tensors, matrices, and figures are excluded from version control.

## Installation

Clone the repository:

```bash
git clone https://github.com/jsharma0110/llm-representation-alignment.git
cd llm-representation-alignment
```

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install the alignment package in editable mode:

```bash
pip install -e .
```

Verify the installation:

```bash
python -c "import alignment; print('alignment package installed')"
```

Some gated Hugging Face models, including Llama models, may require authentication:

```bash
huggingface-cli login
```

## Running Experiments

Run all commands from the repository root.

### TinyLlama vs. Qwen

Extract hidden-state representations:

```bash
python experiments/llm/tinyllama_qwen/run_hidden_states.py
```

Compute layer-wise Linear CKA:

```bash
python experiments/llm/tinyllama_qwen/run_cka.py
```

Generate the CKA heatmap:

```bash
python experiments/llm/tinyllama_qwen/plot_cka.py
```

### Llama-3.2-1B vs. Llama-3.2-3B

Extract hidden-state representations:

```bash
python experiments/llm/llama1b_llama3b/run_hidden_states.py
```

Compute Linear CKA:

```bash
python experiments/llm/llama1b_llama3b/run_cka.py
```

Generate the CKA heatmap:

```bash
python experiments/llm/llama1b_llama3b/plot_cka.py
```

Compute layer-wise Procrustes similarity:

```bash
python experiments/llm/llama1b_llama3b/run_procrustes.py
```

Generate the Procrustes heatmap:

```bash
python experiments/llm/llama1b_llama3b/plot_procrustes.py
```

Run held-out Procrustes analysis with a shuffled-prompt control:

```bash
python experiments/llm/llama1b_llama3b/run_procrustes_heldout.py
```

Visualize the held-out analysis:

```bash
python experiments/llm/llama1b_llama3b/plot_procrustes_heldout.py
```

## Package Components

### `alignment.datasets`

Dataset loading and preprocessing utilities.

Currently includes support for loading TruthfulQA prompts.

### `alignment.models`

Reusable model and tokenizer loading utilities for Hugging Face causal language models.

### `alignment.extraction`

Hidden-state extraction and token-pooling utilities.

### `alignment.similarity`

Representation comparison methods including:

- Linear Centered Kernel Alignment (CKA)
- Orthogonal Procrustes similarity
- held-out Procrustes alignment

### `alignment.visualization`

Reusable visualization utilities for similarity matrices and held-out alignment analyses.

### `alignment.benchmarking`

Infrastructure for model inference and performance benchmarking.

## Datasets

Current datasets include:

- TruthfulQA, loaded dynamically using the Hugging Face `datasets` library
- a small factual QA dataset under `data/factual_qa/`

## Generated Outputs

Hidden-state tensors are written to:

```text
results/hidden_states/
```

Pairwise analysis results are written to:

```text
results/tinyllama_qwen/
results/llama1b_llama3b/
```

Plots are written to:

```text
figures/tinyllama_qwen/
figures/llama1b_llama3b/
```

Generated experiment artifacts are intentionally excluded from Git.

## Development

Reusable functionality should be implemented inside:

```text
src/alignment/
```

Experiment-specific configuration and orchestration should remain inside:

```text
experiments/
```

Before committing changes, verify that the package and experiment scripts compile:

```bash
python -m compileall src/alignment experiments
```

## Contributors

- Jahnavi Sharma
- Palaash Bhathena
- Ismail Jamal
- Krishna Praneet Gudipaty, research supervisor

University of Massachusetts Amherst