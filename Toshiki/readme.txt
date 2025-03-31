Do the following to install packages
!pip install -U git+https://github.com/huggingface/transformers.git
!pip install datasets
!pip install mecab-python3 unidic-lite
!pip install konlpy
!pip install -U bitsandbytes transformers peft accelerate trl datasets sentencepiece wandb
!pip install flash-attn --no-build-isolation
!pip install unsloth
!pip install tf-keras
!pip install torchvision

in logitlens directory, early_decoding.py calculates the language probability (use it with a dataset in data directory).
however, it does not consider a whitespace problem (for langauges that separate words with a whitespace, early_decoding.py does not match target words in the dataset because there is a whitespace usually in the internal model predictions).
For this reason, you should take out the top_k_dict.json that can be obtained after the execution of a code, and recalculate using a function that is in the notebook.
since the probability threshold is p>0.1 and top_k_dict.json has top 10 predictions for each layer for all datapoints, mathematically there is no problem in recomputing the plots from top_k_dict.json

logit lens graph recalculation and statistical testing (some are written with the help of chatgpt and gemini)
https://colab.research.google.com/drive/1Y_bUeLt-zYyKY5Tl-T9b-ARD0wFdm7Xw?usp=sharing

for datasets, refer to readme.txt in data repository.