# Neuron Specialization in Multilingual LLMs

This folder contains tools for investigating neuron specialization in the AYA-23 multilingual language model across different language pairs.

## Overview

This project analyzes neuron activations in the AYA-23-8B model during translation tasks across 80 language pairs. The goal is to identify neurons that specialize in specific languages or language families, providing insights into how multilingual models represent different languages internally.

## Requirements

- Python 3.8+
- PyTorch
- Transformers (v4.49.0)
- datasets
- tiktoken
- sentencepiece
- protobuf
- tqdm
- Hugging Face account with API token

## Setup

1. Clone this repository
2. Install the required dependencies:
```bash
pip install datasets tiktoken sentencepiece protobuf
pip install transformers==4.49.0
```

## Scripts

### Main Scripts

- **aya_get_neurons.py**: Collects neuron activations from the AYA-23-8B model for specific language pairs
- **run_activation_collection.sh**: Batch script for collecting activations across all language pairs

### Usage

To run the activation collection for all language pairs:

```bash
./run_activation_collection.sh
```

To run for a specific language pair:

```bash
python aya_get_neurons.py \
    --model_name "CohereForAI/aya-23-8B" \
    --save_path "/path/to/save/directory" \
    --language_pair "en-fr" \
    --split "validation" \
    --max_samples 30000 \
    --fp16
```

## Language Pairs

The scripts process 80 language pairs organized by resource availability:

- **High resource (5M+)**: de, nl, fr, es, ru, cs, hi, bn, ar, he (with English)
- **Medium resource (1M)**: sv, da, it, pt, pl, bg, kn, mr, mt, ha (with English)
- **Low resource (100k)**: af, lb, ro, oc, uk, sr, sd, gu, ti, am (with English)
- **Extremely low resource (50k)**: no, is, ast, ca, be, bs, ne, ur, kab, so (with English)

Each language pair is processed in both directions (e.g., en-fr and fr-en).

## Data

The project uses the EC40 dataset from Hugging Face for translation examples. Activations are collected from the MLP down-projection layers in each transformer block.

## Output

Activation data is saved as pickle files in the specified output directory, organized by language pair. The results include counts of non-zero activations for each neuron across all processed examples.
All activation (.pkl) files can be found in the following directory: `all-exp-s/Aya/activations_aya-23-8B`

## Analysis

After collecting activations, further analysis can be performed to:
- Identify neurons with high activation for specific languages
- Compare activation patterns across language families
- Correlate neuron activity with translation performance
- Visualize neuron specialization across the model

## Notes

- Processing all language pairs requires significant computational resources and time
- For large models, consider using the `--fp16` flag to reduce memory usage
- The `PYTORCH_CUDA_ALLOC_CONF` environment variable is set to manage GPU memory fragmentation
