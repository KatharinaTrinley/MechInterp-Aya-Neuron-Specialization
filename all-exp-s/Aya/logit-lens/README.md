# Logit Lens Analysis for Multilingual Translation

This folder contains tools and results for analyzing the internal representations of the AYA-23-8B multilingual language model during translation tasks using the Logit Lens technique.

## Overview

The Logit Lens technique allows us to inspect the internal representations at each layer of a transformer model by projecting hidden states directly to the vocabulary space. This provides insights into how the model's "understanding" of the target translation develops through the network layers.

## Contents

- `logit_lens.py`: The main script for performing Logit Lens analysis
- `logit_lens_results/`: Folder containing analysis results
  - CSV files: Token predictions by layer for various language pairs
  - PNG visualizations: Heatmap visualizations of entropy and token predictions

## How It Works

The script:
1. Takes an English sentence and translates it to multiple target languages
2. For each layer in the model, projects the hidden states to token predictions
3. Tracks how these predictions change through the layers
4. Visualizes the results as heatmaps with token annotations
5. Calculates entropy to measure prediction certainty at each position and layer

## Usage

To run the analysis:

```bash
python logit_lens.py
```

The script analyzes translations of "The beautiful flower blooms in the garden." into languages from five language families:

- **Germanic**: German, Afrikaans, Norwegian
- **Romance**: French, Occitan, Catalan
- **Slavic**: Ukrainian, Bulgarian, Serbian
- **Indic**: Hindi, Marathi, Kannada
- **Semitic/Afroasiatic**: Arabic, Somali, Maltese

## Visualizations

The script generates two types of visualizations for each language:

1. **Standard visualization**: A heatmap showing entropy values with token annotations for all output tokens
2. **Wendler-style visualization**: An alternative visualization format inspired by Wendler et al.'s work, focusing on the last few tokens

## Results

The `logit_lens_results/` directory contains:

- CSV files with layer-by-token predictions for both high-resource languages (HRL) and low-resource languages (LRL)
- PNG visualizations highlighting how token predictions develop through network layers

### Example Visualizations

Below are key visualizations from different language families:

#### Germanic Languages
![German Logit Lens](all-exp-s/Aya/logit-lens/results/german_wendler.png)
![Afrikaans Logit Lens](all-exp-s/Aya/logit-lens/results/afrikaans_wendler.png)

#### Slavic Languages
![Bulgarian Logit Lens](all-exp-s/Aya/logit-lens/results/bulgarian_wendler.png)
![Ukrainian Logit Lens](all-exp-s/Aya/logit-lens/results/ukrainian_wendler.png)

These visualizations show how token predictions evolve through the model layers, with colors representing normalized entropy (uncertainty) at each position. Note the differences in prediction stability between high-resource languages (like German) and lower-resource languages.

## Interpretation

These visualizations reveal:

- How early or late in the network specific tokens are "resolved"
- Which language tokens require more processing depth
- Differences in model certainty between high and low resource languages
- Token prediction stability across layers

## Dependencies

- PyTorch
- Transformers
- NumPy
- Pandas
- Matplotlib
- Seaborn
- tqdm

## References

This implementation is inspired by research on model interpretability, including:
- Nostalgebraist's original Logit Lens
- Geva et al. (2022) "Transformer Feed-Forward Layers Are Key-Value Memories"
- Wendler et al. (2023) "LLMs as Pachinko Machines: Understanding the Randomness of Token Prediction for Auto-regressive LLMs"
