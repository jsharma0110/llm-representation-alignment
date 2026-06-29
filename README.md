# LLM Representation Alignment

## Project Goal

This project investigates whether representation alignment and model stitching techniques can be applied to Large Language Models (LLMs). Specifically, we study whether hidden-state representations can help explain differences in factual knowledge, hallucinations, and reasoning ability across models.

---

## Models

- TinyLlama-1.1B-Chat-v1.0
- Qwen2.5-0.5B

Both models are loaded from Hugging Face Transformers.

---

## Dataset

- TruthfulQA
- Small custom factual QA dataset for initial sanity checks

---

## Repository Structure

```
.
├── benchmark_test.py      # Loads TruthfulQA benchmark
├── compare_models.py      # Extract hidden representations from multiple models
├── hidden_states.py       # Hidden-state extraction example
├── evaluate.py            # Runs simple factual QA evaluation
├── factual_qa.json        # Small handcrafted evaluation dataset
└── README.md
```

---

## Current Progress

✅ Installed local Hugging Face and Ollama environments

✅ Downloaded TinyLlama and Qwen models

✅ Built a simple factual QA evaluation pipeline

✅ Loaded the TruthfulQA benchmark

✅ Extracted hidden states from all transformer layers

✅ Compared representation dimensions

- TinyLlama hidden dimension: **2048**
- Qwen hidden dimension: **896**

This dimensional mismatch motivates future representation alignment methods.

---

# Reproducing the Experiment

## 1. Clone the repository

```bash
git clone https://huggingface.co/jahnavisharma/llm-representation-alignment
cd llm-representation-alignment
```

## 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install dependencies

```bash
pip install torch transformers datasets
```

(Optional)

```bash
pip install ollama
```

---

## 4. Run the experiments

### Evaluate on custom factual QA

```bash
python evaluate.py
```

---

### Load TruthfulQA

```bash
python benchmark_test.py
```

---

### Extract hidden states

```bash
python hidden_states.py
```

---

### Compare representations across models

```bash
python compare_models.py
```

---

## Expected Outputs

### evaluate.py

- Model responses
- Simple factual accuracy

### benchmark_test.py

- Sample TruthfulQA questions
- Ground-truth answers

### hidden_states.py

- Number of transformer layers
- Hidden-state tensor shapes

Example:

```
Number of layers: 23

First layer:
torch.Size([1, 8, 2048])

Last layer:
torch.Size([1, 8, 2048])
```

---

### compare_models.py

Prints the hidden representation shape for every prompt.

Example:

```
TinyLlama:
torch.Size([4, 2048])

Qwen:
torch.Size([4, 896])
```

---

## Future Work

- Evaluate on the full TruthfulQA benchmark
- Compare hidden representations layer-by-layer
- Align representations from models with different hidden dimensions
- Implement model stitching experiments
- Investigate whether representation similarity correlates with hallucination rates

---

## Authors

Jahnavi Sharma

UMass Amherst

Summer Research Project (Representation Alignment for LLMs)
