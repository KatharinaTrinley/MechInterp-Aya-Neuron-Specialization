from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import datetime
import os
import utils
from huggingface_hub import login
from datasets import load_dataset, Dataset
from tqdm import tqdm
import ast
import pandas as pd
from matplotlib import pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import bitsandbytes as bnb

class Early_Decoding:
    def __init__(self, model_id, mode="standard"):
        self.model_id = model_id
        self.mode = mode

    def load_tokenizer_and_model(self, output_hidden_states=True):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForCausalLM.from_pretrained(self.model_id, output_hidden_states=output_hidden_states)
        return tokenizer, model
    
    def early_decoding_gpt2(self, prompt, top_k=10, write_several = False):
        tokenizer, model = self.load_tokenizer_and_model()
        encoded_input = tokenizer(prompt, return_tensors='pt')

        # Forward pass through the model to capture intermediate predictions
        with torch.no_grad():
            outputs = model(**encoded_input)

        # Extract residual streams (hidden states) after each layer
        hidden_states = outputs.hidden_states
        last_token_position = encoded_input["input_ids"].size(1)-1  # Last token index

        # Decode top 5 predictions after each layer
        intermediate_predictions = []
        for layer_idx, hidden_state in enumerate(hidden_states):
            # Take the hidden state at the last token position
            last_token_hidden_state = hidden_state[:, last_token_position, :]

            # Pass it through the final layer norm and generate logits
            normalized_hidden_state = model.transformer.ln_f(last_token_hidden_state)
            logits = model.lm_head(normalized_hidden_state)

            # Calculate probabilities using softmax
            probabilities = F.softmax(logits, dim=-1)

            # Get the top-k predictions
            top_k_probs, top_k_indices = torch.topk(probabilities, k=top_k, dim=-1)
            top_k_tokens = tokenizer.batch_decode(top_k_indices[0], skip_special_tokens=True)

            # Store layer predictions
            intermediate_predictions.append({
                "layer": layer_idx,
                "predictions": [{"token": token, "probability": prob.item()} for token, prob in zip(top_k_tokens, top_k_probs[0])]
            })
        
        to_write = ""

        # Pretty-print
        for layer_prediction in intermediate_predictions:
            layer = layer_prediction["layer"]
            to_write += f"\nLayer {layer} Predictions:\n"
            for prediction in layer_prediction["predictions"]:
                token = prediction["token"]
                probability = prediction["probability"]
                to_write += f"  Token: '{token}' | Probability: {probability:.4f}\n"

        if write_several == True:
            text = self.write_output(to_write, prompt, write_several = True)
            return text
        else:
            self.write_output(to_write, prompt)

    def early_decoding_xglm(self, prompt, top_k=10, write_several = False):
        tokenizer, model = self.load_tokenizer_and_model()
        encoded_input = tokenizer(prompt, return_tensors='pt')

        # Forward pass through the model to capture intermediate predictions
        with torch.no_grad():
            outputs = model(**encoded_input)

        # Extract residual streams (hidden states) after each layer
        hidden_states = outputs.hidden_states
        last_token_position = encoded_input["input_ids"].size(1)-1  # Last token index

        # Decode top 5 predictions after each layer
        intermediate_predictions = []
        for layer_idx, hidden_state in enumerate(hidden_states):
            # Take the hidden state at the last token position
            last_token_hidden_state = hidden_state[:, last_token_position, :]

            # Pass it through the final layer norm and generate logits
            normalized_hidden_state = model.model.layer_norm(last_token_hidden_state)
            logits = model.lm_head(normalized_hidden_state)

            # Calculate probabilities using softmax
            probabilities = F.softmax(logits, dim=-1)

            # Get the top-k predictions
            top_k_probs, top_k_indices = torch.topk(probabilities, k=top_k, dim=-1)
            top_k_tokens = tokenizer.batch_decode(top_k_indices[0], skip_special_tokens=True)

            # Store layer predictions
            intermediate_predictions.append({
                "layer": layer_idx,
                "predictions": [{"token": token, "probability": prob.item()} for token, prob in zip(top_k_tokens, top_k_probs[0])]
            })
        
        to_write = ""

        # Pretty-print
        for layer_prediction in intermediate_predictions:
            layer = layer_prediction["layer"]
            to_write += f"\nLayer {layer} Predictions:\n"
            for prediction in layer_prediction["predictions"]:
                token = prediction["token"]
                probability = prediction["probability"]
                to_write += f"  Token: '{token}' | Probability: {probability:.4f}\n"

        if write_several == True:
            text = self.write_output(to_write, prompt, write_several = True)
            return text
        else:
            self.write_output(to_write, prompt)

    def early_decoding_aya_deepseek(self, prompt, top_k=10, write_several = False):
        tokenizer, model = self.load_tokenizer_and_model()
        encoded_input = tokenizer(prompt, return_tensors='pt')

        # Forward pass through the model to capture intermediate predictions
        with torch.no_grad():
            outputs = model(**encoded_input)

        # Extract residual streams (hidden states) after each layer
        hidden_states = outputs.hidden_states
        last_token_position = encoded_input["input_ids"].size(1)-1  # Last token index

        # Decode top 5 predictions after each layer
        intermediate_predictions = []
        for layer_idx, hidden_state in enumerate(hidden_states):
            # Take the hidden state at the last token position
            last_token_hidden_state = hidden_state[:, last_token_position, :]

            # Pass it through the final layer norm and generate logits
            normalized_hidden_state = model.model.norm(last_token_hidden_state)
            logits = model.lm_head(normalized_hidden_state)

            # Calculate probabilities using softmax
            probabilities = F.softmax(logits, dim=-1)

            # Get the top-k predictions
            top_k_probs, top_k_indices = torch.topk(probabilities, k=top_k, dim=-1)
            top_k_tokens = tokenizer.batch_decode(top_k_indices[0], skip_special_tokens=True)

            # Store layer predictions
            intermediate_predictions.append({
                "layer": layer_idx,
                "predictions": [{"token": token, "probability": prob.item()} for token, prob in zip(top_k_tokens, top_k_probs[0])]
            })
        
        to_write = ""

        # Pretty-print
        for layer_prediction in intermediate_predictions:
            layer = layer_prediction["layer"]
            to_write += f"\nLayer {layer} Predictions:\n"
            for prediction in layer_prediction["predictions"]:
                token = prediction["token"]
                probability = prediction["probability"]
                to_write += f"  Token: '{token}' | Probability: {probability:.4f}\n"

        if write_several == True:
            text = self.write_output(to_write, prompt, write_several = True)
            return text
        else:
            self.write_output(to_write, prompt)

    def aya_logit_lens_experiment(self, json_path, task_name, top_k=10):
        #load the data
        ds = load_dataset("json", data_files=json_path)

        #create toy data
        ds = ds["train"]

        #load translation data and create a dictionary
        ds_trans = load_dataset("csv", data_files="word_translation2.csv")
        
        #preparing translation dict
        lang_list = ['fr', 'de', 'ru', 'en', 'zh', 'es', 'ja', 'ko', 'et', 'fi', 'nl', 'hi', 'it']
        D_trans = {}
        for e in ds_trans["train"]:
            for l in lang_list:
                try: e[l] = ast.literal_eval(e[l])
                except:
                    L = []
                    L.append(e[l])
            D_trans[e["word_original"]] = e

        '''
        tokenizer = AutoTokenizer.from_pretrained("CohereForAI/aya-23-8B")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            "CohereForAI/aya-23-8B",
            quantization_config=quantization_config,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            output_hidden_states=True,
        )
        '''
        tokenizer, model = self.load_tokenizer_and_model()
        if self.mode == "baseline":
            model = PeftModel.from_pretrained(model, "tona3738/aya_qlora_wo_codemixing_10Mtokens")
        if self.mode == "codemixing_model":
            model = PeftModel.from_pretrained(model, "tona3738/aya_qlora_with_codemixing_10Mtokens")

        to_write = ""
        to_write2 = ""
        Prompt2topkpred = {}
        Prompt2targetwords = {}
        Json_class = utils.Utils()

        for d in tqdm(ds):
            to_write += "prompt: " + d["prompt"] + "\n\n"
            to_write2 += "prompt: " + d["prompt"] + "\n\n"
            encoded_input = tokenizer(d["prompt"], return_tensors='pt')

            # Forward pass through the model to capture intermediate predictions
            with torch.no_grad():
                outputs = model(**encoded_input)

            # Extract residual streams (hidden states) after each layer
            hidden_states = outputs.hidden_states
            last_token_position = encoded_input["input_ids"].size(1)-1  # Last token index

            # Decode top 5 predictions after each layer
            intermediate_predictions = []
            Layer_and_lang = []
            for layer_idx, hidden_state in enumerate(hidden_states):
                to_write2 += "layer:" + str(layer_idx) + "\n"
                # Take the hidden state at the last token position
                last_token_hidden_state = hidden_state[:, last_token_position, :]

                # Pass it through the final layer norm and generate logits
                if (self.mode == "baseline") or (self.mode == "codemixing_model"):
                    normalized_hidden_state = model.base_model.model.model.norm(last_token_hidden_state)
                else:
                    normalized_hidden_state = model.model.norm(last_token_hidden_state)
                logits = model.lm_head(normalized_hidden_state)

                # Calculate probabilities using softmax
                probabilities = F.softmax(logits, dim=-1)

                # Get the top-k predictions
                
                top_k_probs, top_k_indices = torch.topk(probabilities, k=top_k, dim=-1)
                top_k_tokens = tokenizer.batch_decode(top_k_indices[0], skip_special_tokens=True)
                
                #print("debug:",probabilities.size())

                last_token_probs = probabilities[0, :]

                word_original = d["word_original"]
                error_number = 0
                try:
                    Lang2word = {}
                    for l in lang_list:
                        #print("language:", l)
                        word_list = D_trans[word_original][l]
                        Word2prob = {}
                        for w in word_list:
                            #print("word:", w)
                            to_write2 += "word:" + w + "|"
                            target_token = w
                            tokens = tokenizer.tokenize(target_token)
                            token_ids = tokenizer.convert_tokens_to_ids(tokens)
                            target_token_id = token_ids[0]
                            #print("token id measured:",token_ids[0])
                            token_measured = tokenizer.decode(token_ids[0])[0]
                            to_write2 += "token:" + token_measured + "|" + tokens[0] + "|"
                            #print("token measured:", token_measured)
                            target_token_prob = last_token_probs[target_token_id].item()
                            #print("probability:",target_token_prob)
                            to_write2 += "probability:" + str(target_token_prob) + "\n"
                            #print()
                            #print()
                            Word2prob[w] = (token_measured, target_token_prob)
                        Lang2word[l] = Word2prob
                except:
                    print("error", d)
                    error_number += 1
                print("number of errors:",error_number)

                to_write2 += "\n"

                Layer_and_lang.append({
                    "layer": layer_idx,
                    "predictions": Lang2word
                })

                # Store layer predictions
                intermediate_predictions.append({
                    "layer": layer_idx,
                    "predictions": [{"token": token, "probability": prob.item()} for token, prob in zip(top_k_tokens, top_k_probs[0])]
                })
        
            # Pretty-print
            for layer_prediction in intermediate_predictions:
                layer = layer_prediction["layer"]
                to_write += f"\nLayer {layer} Predictions:\n"
                for prediction in layer_prediction["predictions"]:
                    token = prediction["token"]
                    probability = prediction["probability"]
                    to_write += f"  Token: '{token}' | Probability: {probability:.4f}\n"
            to_write += "\n\n"
            to_write2 += "\n\n"
            
            os.makedirs(f"{self.mode}{task_name}_backup_output", exist_ok=True)
            time = datetime.datetime.now()
            filename = f'./{self.mode}{task_name}_backup_output/topk_log' + time.strftime('%Y%m%d_%H%M%S') + '.json'
            filename2 = f'./{self.mode}{task_name}_backup_output/target_log' + time.strftime('%Y%m%d_%H%M%S') + '.json'
            Prompt2topkpred[d["prompt"]] = intermediate_predictions
            Prompt2targetwords[d["prompt"]] = {"word_original":d["word_original"], "Layer_and_lang":Layer_and_lang}
            Json_class.write_json(filename, Prompt2topkpred)
            Json_class.write_json(filename2, Prompt2targetwords)

        os.makedirs(f"{self.mode}{task_name}_final_output", exist_ok=True)
        filename3 = f'./{self.mode}{task_name}_final_output/topk_dict.json'
        Json_class.write_json(filename3, Prompt2topkpred)
        filename4 = f'./{self.mode}{task_name}_final_output/target_dict.json'
        Json_class.write_json(filename4, Prompt2targetwords)
        self.write_txt(to_write, f"./{self.mode}{task_name}_final_output/logit_lens_topk.txt")
        self.write_txt(to_write2, f"./{self.mode}{task_name}_final_output/logit_lens_target.txt")


        for prompt in Prompt2targetwords:
            try:
                layer_list = Prompt2targetwords[prompt]["Layer_and_lang"]
                word_original = Prompt2targetwords[prompt]["word_original"]
                layers = []
                for i in layer_list:
                    layer_idx = i["layer"]
                    lang_dict = i["predictions"]
                    total_probs = []
                    for lang in lang_list:
                        total_prob = self.calculate_total_prob(lang,lang_dict)
                        total_probs.append(total_prob)
                    layers.append(total_probs)
                self.plot_and_save(layers, lang_list, word_original, task_name)
            except:
                print("error with the prompt:", prompt)

        '''
        if write_several == True:
            text = self.write_output(to_write, prompt, write_several = True)
            return text
        else:
            self.write_output(to_write, prompt)
        '''
    
    def calculate_total_prob(self, lang, lang_dict):
        word_dict = lang_dict[lang]
        total_prob = 0
        for j in word_dict:
            total_prob += word_dict[j][1]
        return total_prob

    def plot_and_save(self, layers, lang_list, word_original, task_name):
        os.makedirs(f"{self.mode}{task_name}_final_output", exist_ok=True)
        df = pd.DataFrame(layers, columns=lang_list)
        ax = df.plot()
        ax.set_title(f"{word_original}")
        ax.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', borderaxespad=0., ncol=2)
        plt.savefig(f'{self.mode}{task_name}_final_output/{word_original}.png', bbox_inches='tight')
        plt.close()

    def write_txt(self, to_write, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(to_write)

    def write_output(self, to_write, prompt, write_several = False):
        text = ""
        text += "model_id: " + self.model_id + "\n"
        text += "prompt: " + prompt + "\n\n"
        text += to_write
        os.makedirs("output", exist_ok=True)
        time = datetime.datetime.now()
        if self.model_id.find("/") != -1:
            model_name = self.model_id[self.model_id.find("/")+1:]
        else:
            model_name = self.model_id
        filename = './output/log_' + model_name + time.strftime('%Y%m%d_%H%M%S') + '.txt'
        if write_several == True:
            return text
        else:
            with open(filename, mode="w",encoding="utf-8") as f:
                f.write(text)
    
    def get_model_detail(self):
        '''
        this function is used to print the model detail
        '''
        tokenizer, model = self.load_tokenizer_and_model(self.model_id)
        model_architecture = str(model)
        os.makedirs("output", exist_ok=True)
        if self.model_id.find("/") != -1:
            model_name = self.model_id[self.model_id.find("/")+1:]
        else:
            model_name = self.model_id
        filename = './output/' + model_name + '_architecture.txt'
        with open(filename, mode="w",encoding="utf-8") as f:
            f.write(model_architecture)
    
    def several_prompts(self, prompt_file_path, which_func):
        Json_class = utils.Utils()
        L = Json_class.load_json(prompt_file_path)
        to_write = ""
        if which_func == "aya_deepseek":
            for i in L:
                text = self.early_decoding_aya_deepseek(i, write_several=True)
                to_write += text + "\n\n"
        elif which_func == "gpt2":
            for i in L:
                text = self.early_decoding_gpt2(i, write_several=True)
                to_write += text + "\n\n"
        elif which_func == "xglm":
            for i in L:
                text = self.early_decoding_xglm(i, write_several=True)
                to_write += text + "\n\n"
        os.makedirs("output", exist_ok=True)
        time = datetime.datetime.now()
        if self.model_id.find("/") != -1:
            model_name = self.model_id[self.model_id.find("/")+1:]
        else:
            model_name = self.model_id
        filename = './output/several_log_' + model_name + time.strftime('%Y%m%d_%H%M%S') + '.txt'
        with open(filename, mode="w",encoding="utf-8") as f:
            f.write(to_write)

if __name__ == "__main__":
    huggingface_token = "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"
    login(huggingface_token)
    #json_path = "./cloze_task_ja.json"
    #json_path2 = "./cloze_task_ko.json"
    #json_path3 = "./translation2_ja2ko.json"
    #json_path4 = "./translation2_ko2ja.json"

    #prompt_file_path = "test.json"
    model_id = "CohereForAI/aya-23-8B"
    #model_id = "meta-llama/Llama-3.1-8B"
    #model_id = "gpt2-medium"
    #model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    #model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    #model_id = "facebook/xglm-564M"
    #early_decoding.early_decoding_gpt2(text)
    #early_decoding.early_decoding_xglm(text)
    #early_decoding.early_decoding_aya_deepseek(text)
    #early_decoding.several_prompts(prompt_file_path, "gpt2")
    #early_decoding.get_model_detail(model_id)

    #task_name = "cloze_ja"
    #early_decoding = Early_Decoding(model_id)
    #early_decoding.aya_logit_lens_experiment(json_path, task_name)
    '''
    task_name = "cloze_ja"
    early_decoding = Early_Decoding(model_id, mode="baseline")
    early_decoding.aya_logit_lens_experiment(json_path, task_name)
    task_name = "cloze_ko"
    early_decoding = Early_Decoding(model_id)
    early_decoding.aya_logit_lens_experiment(json_path2, task_name)
    task_name = "cloze_ko"
    early_decoding = Early_Decoding(model_id, mode="baseline")
    early_decoding.aya_logit_lens_experiment(json_path2, task_name)
    task_name = "transja2ko"
    early_decoding = Early_Decoding(model_id)
    early_decoding.aya_logit_lens_experiment(json_path3, task_name)
    task_name = "transja2ko"
    early_decoding = Early_Decoding(model_id, mode="baseline")
    early_decoding.aya_logit_lens_experiment(json_path3, task_name)
    task_name = "transko2ja"
    early_decoding = Early_Decoding(model_id)
    early_decoding.aya_logit_lens_experiment(json_path4, task_name)
    task_name = "transko2ja"
    early_decoding = Early_Decoding(model_id, mode="baseline")
    early_decoding.aya_logit_lens_experiment(json_path4, task_name)
    '''
    '''
    text = """prompt: ___은 종이 등에 잉크를 묻혀 글씨를 쓰거나 그림을 그리는 필기구이다. 답: 펜。
___는 열차가 주행하는 통로인 궤도 및 침목, 도상이라고 불리는 자갈 및 그것들을 지탱하는 노반 등의 구조물을 말한다. 답: 철도 궤도。
기하학에서 ___이란 교차하는 두 직선이 한 점에서 만날 때 생기는 한 쌍의 교각 중 서로 이웃하지 않는 것을 뜻한다. 답: """
    early_decoding = Early_Decoding(model_id)
    early_decoding.early_decoding_aya_deepseek(text)
    '''
    '''
    early_decoding = Early_Decoding(model_id)
    early_decoding.aya_logit_lens_experiment(json_path2, "cloze_ko_debugged")
    '''
    '''
    task_name = "transja2it"
    json_path = "translation2_ja2it.json"
    early_decoding = Early_Decoding(model_id)
    early_decoding.aya_logit_lens_experiment(json_path, task_name)

    task_name = "transko2it"
    json_path2 = "translation2_ko2it.json"
    early_decoding = Early_Decoding(model_id)
    early_decoding.aya_logit_lens_experiment(json_path2, task_name)
    '''
    #task_name = "transen2zh_aya"
    #json_path3 = "translation2_en2zh.json"
    #early_decoding = Early_Decoding(model_id)
    #early_decoding.aya_logit_lens_experiment(json_path3, task_name)

    '''
    task_name = "transja2ko_codemixing"
    json_path = "translation2_ja2ko.json"
    early_decoding = Early_Decoding(model_id, mode="codemixing_model")
    early_decoding.aya_logit_lens_experiment(json_path, task_name)

    task_name = "transko2ja_codemixing"
    json_path2 = "translation2_ko2ja.json"
    early_decoding = Early_Decoding(model_id, mode="codemixing_model")
    early_decoding.aya_logit_lens_experiment(json_path2, task_name)

    
    task_name = "cloze_ja_codemixing"
    json_path3 = "cloze_task_ja.json"
    early_decoding = Early_Decoding(model_id, mode="codemixing_model")
    early_decoding.aya_logit_lens_experiment(json_path3, task_name)

    task_name = "cloze_ko_codemixing"
    json_path4 = "cloze_task_ko_debugged.json"
    early_decoding = Early_Decoding(model_id, mode="codemixing_model")
    early_decoding.aya_logit_lens_experiment(json_path4, task_name)
    '''

    #task_name = "cloze_ko_baseline"
    #json_path5 = "cloze_task_ko_debugged.json"
    #early_decoding = Early_Decoding(model_id, mode="baseline")
    #early_decoding.aya_logit_lens_experiment(json_path5, task_name)

    task_name = "mixcloze_ja_standard"
    json_path = "codemixing_cloze_task_ja.json"
    early_decoding = Early_Decoding(model_id)
    early_decoding.aya_logit_lens_experiment(json_path, task_name)

    task_name2 = "mixcloze_ko_standard"
    json_path2 = "codemixing_cloze_task_ko.json"
    early_decoding = Early_Decoding(model_id)
    early_decoding.aya_logit_lens_experiment(json_path2, task_name2)
    '''
    task_name3 = "mixcloze_ja_baseline"
    json_path3 = "codemixing_cloze_task_ja.json"
    early_decoding = Early_Decoding(model_id, mode="baseline")
    early_decoding.aya_logit_lens_experiment(json_path3, task_name3)

    task_name4 = "mixcloze_ko_baseline"
    json_path4 = "codemixing_cloze_task_ko.json"
    early_decoding = Early_Decoding(model_id, mode="baseline")
    early_decoding.aya_logit_lens_experiment(json_path4, task_name4)

    task_name5 = "mixcloze_ja_codemixing"
    json_path5 = "codemixing_cloze_task_ja.json"
    early_decoding = Early_Decoding(model_id, mode="codemixing_model")
    early_decoding.aya_logit_lens_experiment(json_path5, task_name5)

    task_name6 = "mixcloze_ko_codemixing"
    json_path6 = "codemixing_cloze_task_ko.json"
    early_decoding = Early_Decoding(model_id, mode="codemixing_model")
    early_decoding.aya_logit_lens_experiment(json_path6, task_name6)
    '''
    #model_id2 = "meta-llama/Llama-3.1-8B"
    #task_name = "transen2zh"
    #json_path4 = "translation2_en2zh.json"
    #early_decoding = Early_Decoding(model_id2)
    #early_decoding.aya_logit_lens_experiment(json_path4, task_name)
    """
    try:
        early_decoding = Early_Decoding(model_id)
        early_decoding.aya_logit_lens_experiment(json_path2)
    except:
        print("standard error 2")
    try:
        early_decoding = Early_Decoding(model_id, mode="baseline")
        early_decoding.aya_logit_lens_experiment(json_path2)
    except:
        print("baseline error 2")
    """
    '''
    model_id2 = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    early_decoding = Early_Decoding(model_id2)
    early_decoding.early_decoding_aya_deepseek(text)
    '''