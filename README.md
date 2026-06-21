# LLM Representation Alignment

## Project Goal
Investigate whether representation alignment and stitching techniques can be applied to LLMs and whether hidden-state similarities can help explain hallucinations and factual errors.

## Models
- TinyLlama-1.1B-Chat-v1.0
- Qwen2.5-0.5B

## Dataset
- TruthfulQA

## Current Progress
- Set up local Hugging Face environment
- Downloaded and ran TinyLlama and Qwen
- Built factual QA evaluation pipeline
- Evaluated models on TruthfulQA questions
- Extracted hidden states from all transformer layers
- Compared representation dimensions
  - TinyLlama: 2048
  - Qwen: 896
- Identified dimensional mismatch as an initial alignment challenge

## Future Work
- Evaluate larger TruthfulQA subsets
- Compare representations across layers
- Implement alignment between models with different hidden dimensions
- Explore stitching methods
- Investigate relationships between representation similarity and hallucinations
