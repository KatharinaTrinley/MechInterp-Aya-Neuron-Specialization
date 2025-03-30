#!/bin/bash

pip install datasets tiktoken sentencepiece protobuf
pip install transformers==4.49.0

export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:1028'

SAVE_BASE_DIR="/netscratch/ktrinley/MechInterp-Project/Aya/activations_aya-23-8B"
MODEL_NAME="CohereForAI/aya-23-8B"
# validation split
SPLIT="validation"
MAX_SAMPLES=30000
FP16_FLAG="--fp16"

mkdir -p $SAVE_BASE_DIR

# All EC40 language pairs (both directions with English)
PAIRS=(
    # High resource (5M+): de, nl, fr, es, ru, cs, hi, bn, ar, he
    'en-de' 'en-nl' 'en-fr' 'en-es' 'en-ru' 'en-cs' 'en-hi' 'en-bn' 'en-ar' 'en-he'
    'de-en' 'nl-en' 'fr-en' 'es-en' 'ru-en' 'cs-en' 'hi-en' 'bn-en' 'ar-en' 'he-en'
    
    # Medium resource (1M): sv, da, it, pt, pl, bg, kn, mr, mt, ha
    'en-sv' 'en-da' 'en-it' 'en-pt' 'en-pl' 'en-bg' 'en-kn' 'en-mr' 'en-mt' 'en-ha'
    'sv-en' 'da-en' 'it-en' 'pt-en' 'pl-en' 'bg-en' 'kn-en' 'mr-en' 'mt-en' 'ha-en'
    
    # Low resource (100k): af, lb, ro, oc, uk, sr, sd, gu, ti, am
    'en-af' 'en-lb' 'en-ro' 'en-oc' 'en-uk' 'en-sr' 'en-sd' 'en-gu' 'en-ti' 'en-am'
    'af-en' 'lb-en' 'ro-en' 'oc-en' 'uk-en' 'sr-en' 'sd-en' 'gu-en' 'ti-en' 'am-en'
    
    # Extremely low resource (50k): no, is, ast, ca, be, bs, ne, ur, kab, so
    'en-no' 'en-is' 'en-ast' 'en-ca' 'en-be' 'en-bs' 'en-ne' 'en-ur' 'en-kab' 'en-so'
    'no-en' 'is-en' 'ast-en' 'ca-en' 'be-en' 'bs-en' 'ne-en' 'ur-en' 'kab-en' 'so-en'
)

# all output can be found here in the log file
LOG_FILE="activations_collection.log"

echo "Starting neuron activation collection for ${#PAIRS[@]} language pairs" | tee -a $LOG_FILE
echo "Results will be saved to: $SAVE_BASE_DIR" | tee -a $LOG_FILE
echo "=====================================================" | tee -a $LOG_FILE

# we loop through all the pairs and run the script for each one
for pair in "${PAIRS[@]}"; do
    echo "$(date): Processing language pair: $pair" | tee -a $LOG_FILE
    
    # Run the script for this language pair
    python aya_get_neurons.py \
        --model_name "$MODEL_NAME" \
        --save_path "$SAVE_BASE_DIR" \
        --language_pair "$pair" \
        --split "$SPLIT" \
        --max_samples "$MAX_SAMPLES" \
        $FP16_FLAG 2>&1 | tee -a "$LOG_FILE"
    
    # error handling
    if [ $? -eq 0 ]; then
        echo "$(date): Successfully processed $pair" | tee -a $LOG_FILE
    else
        echo "$(date): ERROR processing $pair" | tee -a $LOG_FILE
    fi
    
    echo "-------------------------------------------------" | tee -a $LOG_FILE
    sleep 5
done

echo "$(date): All language pairs processed. Check $LOG_FILE for details." | tee -a $LOG_FILE