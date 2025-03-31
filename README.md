# MechInterp-Codemixing (Toshiki's Part)

# Neuron-Specialization (Katharina's Part)

This repository contains code and results for analyzing neuron specialization and internal representations in multilingual language models, focusing on the Aya-23-8B model by CohereAI and mBart.

## Overview

The project explores how neurons in large multilingual models specialize in different languages and language families. We use a combination of techniques:

1. **Neuron Activation Analysis**: Tracking which neurons activate for specific language pairs
2. **Logit Lens Analysis**: Examining how token predictions develop through model layers
3. **Cross-lingual Transfer**: Investigating how models handle both high and low-resource languages

## Repository Structure

```
├── Aya/
│   ├── activations_aya-23-8B/      # Neuron activation data for all language pairs
│   │   ├── af-en/
│   │   │   └── activations.pkl
│   │   ├── am-en/
│   │   │   └── activations.pkl
│   │   └── ...
│   ├── logit-lens/                 # Logit lens analysis
│   │   ├── logit_lens_results/
│   │   │   ├── afrikaans_token_matrix.csv
│   │   │   ├── afrikaans_wendler.png
│   │   │   └── ...
│   │   └── logit-lens.py
│   └── Neuron-Specialization/      # Code for neuron activation collection
│       ├── activations_collections.log
│       ├── aya23-NS-EC40.sh
│       ├── aya_get_neurons_EC40.py
│       └── README.md
└── mBart/                         # Similar analysis for mBart model
    ├── 3-mBart_test.sh
    ├── mbart_activations_collection.log
    └── mbart_get_neurons.py
```

## Experimental Setup

### Models

- **Aya-23-8B**: A multilingual decoder-only language model by CohereAI
- **mBart**: A multilingual encoder-decoder model for sequence-to-sequence tasks

### Dataset

We primarily use the EC40 dataset, which contains parallel text across 40 languages with varying resource levels:

| Resource Level  | Languages                                        | Size  |
|----------------|--------------------------------------------------|------|
| High           | de, nl, fr, es, ru, cs, hi, bn, ar, he           | 5M   |
| Medium         | sv, da, it, pt, pl, bg, kn, mr, mt, ha           | 1M   |
| Low            | af, lb, ro, oc, uk, sr, sd, gu, ti, am           | 100k |
| Extremely-Low  | no, is, ast, ca, be, bs, ne, ur, kab, so         | 50k  |

For a detailed list of languages, language families, and overlap between datasets, see the [Language Details](#language-details) section.

## Key Components

### Neuron Activation Collection

The `Neuron-Specialization` directory contains scripts for collecting neuron activations across different language pairs:

- `aya23-NS-EC40.sh`: Batch script for running activation collection on all EC40 language pairs
- `aya_get_neurons_EC40.py`: Python script that records neuron activations during translation tasks

The collected activations are stored as pickle files in the `activations_aya-23-8B` directory, organized by language pair.

### Logit Lens Analysis

The `logit-lens` directory contains tools for analyzing the internal representations of the model:

- `logit-lens.py`: Script implementing the Logit Lens technique to examine layer-wise token predictions
- `logit_lens_results/`: Directory containing analysis results, including:
  - CSV files with token predictions for each layer
  - Visualization plots showing how predictions evolve through the model

## Running the Code

### Neuron Activation Collection

```bash
cd Aya/Neuron-Specialization
./aya23-NS-EC40.sh
```

### Logit Lens Analysis

```bash
cd Aya/logit-lens
python logit-lens.py
```

## Language Details

<details>
<summary>Overlapping Languages Between Aya101 and EC40</summary>

The following table shows the 35 languages that appear in both the Aya101 and EC40 datasets, along with their resourcedness levels in each dataset.

| ISO Code | Language Name  |EC40 Resourcedness | In Aya-23 |
|----------|---------------|-------------------|-----------|
| afr      | Afrikaans     |Low               | No        |
| amh      | Amharic       |Low               | No        |
| ara      | Arabic        | High              | Yes       |
| bel      | Belarusian    | Extremely-Low     | No        |
| ben      | Bengali       |  High              | No        |
| bul      | Bulgarian     |Medium            | No        |
| cat      | Catalan       |  Extremely-Low     | No        |
| ces      | Czech         | High              | Yes       |
| dan      | Danish        |  Medium            | No        |
| deu      | German        |  High              | Yes       |
| fra      | French        | High              | Yes       |
| guj      | Gujarati      |  Low               | No        |
| hau      | Hausa         |Medium            | No        |
| heb      | Hebrew        | High              | Yes       |
| hin      | Hindi         |High              | Yes       |
| isl      | Icelandic     | Extremely-Low     | No        |
| ita      | Italian       | Medium            | Yes       |
| kan      | Kannada       | Medium            | No        |
| ltz      | Luxembourgish | Low               | No        |
| mar      | Marathi       | Medium            | No        |
| mlt      | Maltese       |  Medium            | No        |
| nep      | Nepali        |Extremely-Low     | No        |
| nld      | Dutch         |High              | Yes       |
| nor      | Norwegian     | Extremely-Low     | No        |
| pol      | Polish        |  Medium            | Yes       |
| por      | Portuguese    |  Medium            | Yes       |
| ron      | Romanian      | Low               | Yes       |
| rus      | Russian       |  High              | Yes       |
| snd      | Sindhi        |  Low               | No        |
| som      | Somali        | Extremely-Low     | No        |
| spa      | Spanish       |High              | Yes       |
| srp      | Serbian       | Low               | No        |
| swe      | Swedish       | Medium            | No        |
| ukr      | Ukrainian     |  Low               | Yes       |
| urd      | Urdu          | Extremely-Low     | No        |
</details>

<details>
<summary>Language Families</summary>

| Family       | Languages                                                                                                                                                            |
|-------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Germanic    | German, Dutch, Swedish, Danish, Afrikaans, Luxembourgish, Norwegian, Icelandic, English, Frisian, Faroese, Yiddish, Scots                                            |
| Romance     | French, Spanish, Italian, Portuguese, Romanian, Occitan, Asturian, Catalan, Galician, Corsican, Sicilian, Venetian, Aragonese                                        |
| Slavic      | Russian, Czech, Polish, Bulgarian, Ukrainian, Serbian, Belarusian, Bosnian, Slovak, Slovene, Macedonian, Montenegrin                                                 |
| Indo-Aryan  | Hindi, Bengali, Kannada, Marathi, Sindhi, Gujarati, Nepali, Urdu, Punjabi, Assamese, Sinhala, Konkani, Maithili, Rajasthani, Bhojpuri, Odia                         |
| Afro-Asiatic | Arabic, Hebrew, Maltese, Amharic, Tigrinya, Hausa, Kabyle, Somali, Berber                                                                                          |
</details>

## Key Findings

- Neurons in multilingual models show specialization for specific language families
- Cross-lingual transfer is stronger between related languages
- High-resource languages tend to have more dedicated neurons
- Internal representations (via Logit Lens) reveal how the model's understanding develops through layers

## References

```bibtex
@inproceedings{tan-etal-2024-neuron,
    title = "Neuron Specialization: Leveraging Intrinsic Task Modularity for Multilingual Machine Translation",
    author = "Tan, Shaomu  and
      Wu, Di  and
      Monz, Christof",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.374/",
}
```

```bibtex
@inproceedings{tan2023towards,
  title={Towards a Better Understanding of Variations in Zero-Shot Neural Machine Translation Performance},
  author={Tan, Shaomu and Monz, Christof},
  booktitle={Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
  pages={13553--13568},
  year={2023}
}
```

```bibtex
@inproceedings{tiedemann2012parallel,
  title={Parallel data, tools and interfaces in OPUS.},
  author={Tiedemann, J{\"o}rg},
  booktitle={Lrec},
  volume={2012},
  pages={2214--2218},
  year={2012},
  organization={Citeseer}
}
```

```bibtex
@misc{aryabumi2024aya,
      title={Aya 23: Open Weight Releases to Further Multilingual Progress}, 
      author={Viraat Aryabumi and John Dang and Dwarak Talupuru and Saurabh Dash and David Cairuz and Hangyu Lin and Bharat Venkitesh and Madeline Smith and Kelly Marchisio and Sebastian Ruder and Acyr Locatelli and Julia Kreutzer and Nick Frosst and Phil Blunsom and Marzieh Fadaee and Ahmet Üstün and Sara Hooker},
      year={2024},
      eprint={2405.15032},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

```bibtex
@inproceedings{federmann2022ntrex,
  title={NTREX-128--news test references for MT evaluation of 128 languages},
  author={Federmann, Christian and Kocmi, Tom and Xin, Ying},
  booktitle={Proceedings of the First Workshop on Scaling Up Multilingual Evaluation},
  pages={21--24},
  year={2022}
}
```

```bibtex
@article{costa2022no,
  title={No language left behind: Scaling human-centered machine translation},
  author={Costa-juss{\`a}, Marta R and Cross, James and {\c{C}}elebi, Onur and Elbayad, Maha and Heafield, Kenneth and Heffernan, Kevin and Kalbassi, Elahe and Lam, Janice and Licht, Daniel and Maillard, Jean and others},
  journal={arXiv preprint arXiv:2207.04672},
  year={2022}
}
```
