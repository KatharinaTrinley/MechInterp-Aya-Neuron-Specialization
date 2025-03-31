from datasets import load_dataset
import MeCab
import collections
import datetime
import os
import re
import json
import numpy as np
from tqdm import tqdm
import glob

def if_data_big(DATASET_PATH, lang_split = None, num_iterations = 100):
    os.makedirs("output", exist_ok=True)
    ds = load_dataset(DATASET_PATH, lang_split)
    train_set, test_set = dataset_split(ds["train"]["text"])
    split_idx = np.arange(0, len(train_set), int(len(train_set)/num_iterations))
    for i in tqdm(range(len(split_idx)-1)):
        partial_train_set = train_set[split_idx[i]:split_idx[i+1]]
        total_text_train = wakati_all(partial_train_set)
        word_count_dict_train = word_count(total_text_train)
        write_json(f"./output/japanese_partial{i}.json", word_count_dict_train)
    D = {}
    for path in glob.glob("./output/japanese_partial*.json"):
        small_dict = load_json(path)
        D = D|small_dict
    total_text_test = wakati_all(test_set[0:10000])
    word_count_dict_test = word_count(total_text_test)
    evaluation(D, word_count_dict_test)
    word_count_dict_train = remove_only_one_hiragana_words(D)
    write_json("./japanese_monolingual_dict.json", D)
    sorted_D = sorted(D.items(), key=lambda x:x[1], reverse = True)
    D.clear()
    D.update(sorted_D[0:100000])
    print("the coverage of top 100 thousand words is: ")
    evaluation(D, word_count_dict_test)
    write_json("./japanese_monolingual_dict_topk.json", D)

def if_data_small(DATASET_PATH, write_file_name):
    ds = load_dataset(DATASET_PATH)
    train_set, test_set = dataset_split(ds["train"]["text"][:100])
    total_text_train = wakati_all(train_set)
    total_text_test = wakati_all(test_set)
    word_count_dict_train = word_count(total_text_train)
    word_count_dict_test = word_count(total_text_test)
    evaluation(word_count_dict_train, word_count_dict_test)
    word_count_dict_train = remove_only_one_hiragana_words(word_count_dict_train)
    write_json("./"+write_file_name, word_count_dict_train)

def dataset_split(ds, split_ratio = 0.8):
    train_ratio = int(len(ds)*split_ratio)
    train_set = ds[0:train_ratio]
    test_set = ds[train_ratio:]
    return train_set, test_set

def wakati_all(train_set):
    mecab_wakati = MeCab.Tagger("-Owakati")
    total_text = ""
    for i in train_set:
        total_text += mecab_wakati.parse(i) + " "
    return total_text

def word_count(total_text):
    words = total_text.split()
    word_count_dict = collections.Counter(words)
    return word_count_dict

def evaluation(word_count_dict_train, word_count_dict_test):
    #check the cover rate of the train set
    count = 0
    for i in word_count_dict_train:
        if i in word_count_dict_test:
            count += 1
    cover_rate = count/len(word_count_dict_test)
    vocab_size = len(word_count_dict_train)
    text1 = "cover rate of the train set: " + str(cover_rate) + "\n"
    text2 = "vocab size of the train set: " + str(vocab_size) + "\n"
    text = text1 + text2
    print(text)
    write_log(text)

def write_log(text):
    os.makedirs("output", exist_ok=True)
    time = datetime.datetime.now()
    filename = './output/log_' + time.strftime('%Y%m%d_%H%M%S') + '.txt'
    with open(filename, mode="w",encoding="utf-8") as f:
        f.write(text)

def remove_only_one_hiragana_words(word_count_dict_train):
    hiragana_katakana = re.compile('[\u3041-\u309F]|[\u30A1-\u30FF]')
    to_pop = []
    for w in word_count_dict_train:
        if hiragana_katakana.fullmatch(w):
            to_pop.append(w)
            #print("popped word: ", w)
    for w in to_pop:
        word_count_dict_train.pop(w)
    text = "after removing, the size of the vocab is :" + str(len(word_count_dict_train)) + "\n"
    print(text)
    write_log(text)
    return word_count_dict_train

def write_json(file_name, D):
    with open(file_name, "w") as f:
        d = json.dumps(D)
        f.write(d)

def load_json(file_name):
    d = {}
    with open(file_name, "r") as f:
        d = json.load(f)
    return d

if __name__ == "__main__":
    #DATASET_PATH = "AhmedSSabir/Japanese-wiki-dump-sentence-dataset"
    DATASET_PATH = "wikimedia/wikipedia"
    lang_split = "20231101.ja"
    #lang_split = "20231101.ko"
    if_data_big(DATASET_PATH, lang_split)
