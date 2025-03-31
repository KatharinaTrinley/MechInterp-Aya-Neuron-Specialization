import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer
import datetime
from tqdm import tqdm

class LogitLens:
    # https://huggingface.co/CohereForAI/aya-23-8B
    def __init__(self, model_id="CohereForAI/aya-23-8B"):
        self.model_id = model_id
        
    def load_model(self):
        """Load tokenizer and model"""
        print(f"Loading model {self.model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            return_dict_in_generate=True,
            output_hidden_states=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        model.eval()
        return tokenizer, model
        
    def analyze_translations(self, english_sentence, target_languages=None):
        """Analyze translations with logit lens approach"""
        if target_languages is None:
            target_languages = [
                "German", "Afrikaans", "Norwegian",
                "French", "Occitan", "Catalan",
                "Ukrainian", "Bulgarian", "Serbian",
                "Hindi", "Marathi", "Kannada",
                "Arabic", "Somali", "Maltese"
            ]
        
        # Load model and tokenizer
        tokenizer, model = self.load_model()

        # Create output directory
        os.makedirs("logit_lens_results", exist_ok=True)
        
        # Process each language
        for language in target_languages:
            print(f"Processing translation to {language}...")
            
            # First, generate the translation
            prompt = f"Translate the following English sentence to {language}: \"{english_sentence}\" {language}: "
            
            # Generate and analyze the translation
            self.generate_and_analyze_translation(english_sentence, language, tokenizer, model)
            
        print("Analysis complete. Results saved to the 'logit_lens_results' directory.")
    
    def generate_and_analyze_translation(self, english_sentence, language, tokenizer, model):
        """
        Generate a translation and then analyze it using logit lens
        
        Args:
            english_sentence: The English sentence to translate
            language: Target language
            tokenizer: The model tokenizer
            model: The language model
        """
        # First, create the prompt
        prompt = f"Translate the following English sentence to {language}: \"{english_sentence}\" {language}: "
        
        # Generate the translation
        inputs = tokenizer(prompt, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Set generation parameters - adapted for newer versions
        generation_config = {
            "max_new_tokens": 100,
        }
        
        try:
            # Generate translation
            with torch.no_grad():
                outputs = model.generate(**inputs, **generation_config)
            
            # Fix for the decoding error - ensure we're getting the right format
            if hasattr(outputs, "sequences"):
                # For newer versions that return an object
                generated_ids = outputs.sequences[0]
            else:
                # For older versions that return a tensor directly
                generated_ids = outputs[0]
            
            # Decode the full output
            full_output = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # Extract just the translation part
            translation_start = full_output.rfind(f"{language}:") + len(f"{language}:")
            translation = full_output[translation_start:].strip()
            
            print(f"Generated translation: {translation}")
            
            # Use a simpler approach: create a sentence for analysis by appending 
            # the translation to the prompt
            full_prompt = prompt + translation
            
            return self.analyze_single_translation(full_prompt, language, tokenizer, model)
            
        except Exception as e:
            print(f"Error during generation: {e}")
            print("Falling back to analyzing just the prompt...")
            return self.analyze_single_translation(prompt, language, tokenizer, model)
    
    def analyze_single_translation(self, prompt, language, tokenizer, model, translation=None):
        """Analyze a single translation with direct access to layer outputs"""
        # Tokenize input
        encoded_input = tokenizer(prompt, return_tensors='pt')
        input_ids = encoded_input["input_ids"]
        
        # If CUDA is available, move tensors to GPU
        device = next(model.parameters()).device
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        
        # Forward pass through the model
        with torch.no_grad():
            outputs = model(**encoded_input)
        
        # Get all hidden states
        hidden_states = outputs.hidden_states
        
        # Get the original tokens for reference
        original_tokens = [tokenizer.decode(token_id) for token_id in input_ids[0]]
        
        # Create matrices for storing results
        num_layers = len(hidden_states)
        seq_length = input_ids.size(1)
        
        # Matrix for top token predictions
        token_matrix = np.empty((num_layers, seq_length), dtype=object)
        # Matrix for token probabilities
        prob_matrix = np.zeros((num_layers, seq_length))
        # Matrix for entropy
        entropy_matrix = np.zeros((num_layers, seq_length))
        
        # Access the embedding weights directly
        if hasattr(model, "lm_head"):
            weight = model.lm_head.weight
        else:
            weight = model.get_output_embeddings().weight
        
        # Apply logit lens at each layer
        for layer_idx, hidden_state in enumerate(hidden_states):
            # For each position in the sequence
            for pos in range(seq_length):
                # Get the hidden state for this position
                pos_hidden_state = hidden_state[0, pos].unsqueeze(0)
                
                # Project to vocabulary space to get logits
                # Different models might have different final normalization layers
                try:
                    # Try various model architectures
                    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
                        # GPT-2 style
                        norm_hidden_state = model.transformer.ln_f(pos_hidden_state)
                    elif hasattr(model, "model") and hasattr(model.model, "norm"):
                        # Some newer models (Aya, DeepSeek)
                        norm_hidden_state = model.model.norm(pos_hidden_state)
                    elif hasattr(model, "model") and hasattr(model.model, "layer_norm"):
                        # XGLM style
                        norm_hidden_state = model.model.layer_norm(pos_hidden_state)
                    else:
                        # Fallback - use the hidden state directly
                        norm_hidden_state = pos_hidden_state
                except Exception as e:
                    print(f"Warning: Error in normalization: {e}")
                    norm_hidden_state = pos_hidden_state
                
                # Generate logits
                try:
                    logits = torch.matmul(norm_hidden_state, weight.T)
                except:
                    # Fallback if direct multiplication fails
                    logits = model.lm_head(norm_hidden_state)
                
                # Calculate probabilities
                probs = F.softmax(logits, dim=-1)[0]
                
                # Get top prediction
                top_prob, top_id = probs.max(dim=0)
                top_token = tokenizer.decode(top_id)
                
                # Store results
                token_matrix[layer_idx, pos] = top_token
                prob_matrix[layer_idx, pos] = top_prob.item()
                
                # Calculate entropy
                probs_np = probs.detach().cpu().numpy()
                # Add epsilon to avoid log(0) and filter out zeros before applying log
                non_zero_probs = probs_np[probs_np > 0]
                if len(non_zero_probs) > 0:
                    entropy_matrix[layer_idx, pos] = -np.sum(non_zero_probs * np.log2(non_zero_probs))
                else:
                    entropy_matrix[layer_idx, pos] = 0
        
        # Find where the language token appears at the end of the prompt
        language_token_idx = prompt.rfind(f"{language}:") + len(f"{language}:")
        
        # Find corresponding token index (may span multiple tokens)
        prompt_tokens = tokenizer.encode(prompt[:language_token_idx], add_special_tokens=False)
        output_start_idx = len(prompt_tokens)
        
        # Extract only the output tokens
        output_tokens = original_tokens[output_start_idx:]
        output_token_matrix = token_matrix[:, output_start_idx:]
        output_entropy_matrix = entropy_matrix[:, output_start_idx:]
        
        # Print all output tokens for all layers in a more readable format
        print("\nOutput Token Matrix (Layers × Output Tokens):")
        print(f"Number of layers: {output_token_matrix.shape[0]}")
        print(f"Number of output tokens: {output_token_matrix.shape[1]}")
        print("\nOriginal output tokens:", output_tokens)
        
        # Print the full matrix in a more readable format
        print("\nFull output token matrix:")
        for layer_idx in range(output_token_matrix.shape[0]):
            layer_tokens = output_token_matrix[layer_idx]
            print(f"Layer {layer_idx:2d}: {layer_tokens.tolist()}")
        
        # You can also save this to a CSV file for easier analysis
        import pandas as pd
        df = pd.DataFrame(output_token_matrix)
        # Set column names to token positions
        df.columns = [f"Token_{i}" for i in range(len(output_tokens))]
        # Set row names to layer numbers
        df.index = [f"Layer_{i}" for i in range(output_token_matrix.shape[0])]
        # Save to CSV
        df.to_csv(f"logit_lens_results/{language.lower().replace(' ', '_')}_token_matrix.csv")
        
        # Create standard visualization with only output tokens
        self.create_standard_visualization(
            language, output_tokens, output_token_matrix, output_entropy_matrix, 
            f"logit_lens_results/{language.lower().replace(' ', '_')}_standard.png"
        )
        
        # Create Wendler-style visualization with only output tokens
        self.create_wendler_visualization(
            language, output_tokens, output_token_matrix, output_entropy_matrix,
            f"logit_lens_results/{language.lower().replace(' ', '_')}_wendler.png"
        )
        
        return output_tokens, output_token_matrix, output_entropy_matrix

    def create_standard_visualization(self, language, original_tokens, token_matrix, entropy_matrix, output_file):
        """Create a standard heatmap visualization with token annotations"""
        # we normalize entropy for better visualization
        norm_entropy = (entropy_matrix - entropy_matrix.min()) / (entropy_matrix.max() - entropy_matrix.min() + 1e-10)
        
        plt.figure(figsize=(max(20, len(original_tokens)), 16))
        
        # Heatmap with annotations
        ax = sns.heatmap(
            norm_entropy, 
            cmap="coolwarm", 
            annot=token_matrix,
            fmt="", 
            linewidths=0.5,
            annot_kws={"size": 9, "fontweight": "bold"},
            cbar_kws={"label": "Normalized Entropy"}
        )
        
        # Set labels
        ax.set_title(f"Logit Lens Analysis: English to {language} Translation", fontsize=16)
        ax.set_xlabel("Token Position", fontsize=12)
        ax.set_ylabel("Layer", fontsize=12)
    
        ax.set_xticks(np.arange(len(original_tokens)) + 0.5)
        ax.set_xticklabels(original_tokens, rotation=90, fontsize=10)
    
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Standard visualization saved to {output_file}")
    
    def create_wendler_visualization(self, language, original_tokens, token_matrix, entropy_matrix, output_file):
        """Create a visualization more similar to Wendler et al."""
        # Focus on the last few tokens if sequence is too long
        max_display_tokens = 16
        if len(original_tokens) > max_display_tokens:
            start_idx = max(0, len(original_tokens) - max_display_tokens)
            original_tokens = original_tokens[start_idx:]
            entropy_matrix = entropy_matrix[:, start_idx:]
            token_matrix = token_matrix[:, start_idx:]
        
        norm_entropy = (entropy_matrix - entropy_matrix.min()) / (entropy_matrix.max() - entropy_matrix.min() + 1e-10)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Use imshow (https://www.geeksforgeeks.org/matplotlib-pyplot-imshow-in-python/) instead of heatmap for more direct control
        heatmap = ax.imshow(norm_entropy, cmap='coolwarm', aspect='auto')
        plt.colorbar(heatmap, label="Normalized Entropy")
        
        # add text annotations with their color
        for i in range(norm_entropy.shape[0]):
            for j in range(norm_entropy.shape[1]):
                # Choose text color based on background
                text_color = 'white' if norm_entropy[i, j] > 0.6 or norm_entropy[i, j] < 0.3 else 'black'
                
                # add token to the heatmap
                ax.text(j, i, token_matrix[i, j], ha="center", va="center", 
                        color=text_color, fontsize=8, fontweight='bold')
        
        # Set labels and ticks
        ax.set_title(f"Logit Lens: English to {language}")
        ax.set_xlabel("Token Position")
        ax.set_ylabel("Layer")
        
        ax.set_xticks(np.arange(len(original_tokens)))
        ax.set_xticklabels(original_tokens, rotation=90)
        ax.set_yticks(np.arange(norm_entropy.shape[0]))
        ax.set_yticklabels(np.arange(1, norm_entropy.shape[0] + 1))
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Wendler-style visualization saved to {output_file}")

# Main
if __name__ == "__main__":
    logit_lens = LogitLens("CohereForAI/aya-23-8B")
    
    # english sentence to analyze
    english_sentence = "The beautiful flower blooms in the garden."
    languages = [
        "German", "Afrikaans", "Norwegian",
        "French", "Occitan", "Catalan", 
        "Ukrainian", "Bulgarian", "Serbian",
        "Hindi", "Marathi", "Kannada",
        "Arabic", "Somali", "Maltese"
    ]
    
    # we run the analysis for each language group
    language_groups = {
        "Germanic": ["German", "Afrikaans", "Norwegian"],
        "Romance": ["French", "Occitan", "Catalan"],
        "Slavic": ["Ukrainian", "Bulgarian", "Serbian"],
        "Indic": ["Hindi", "Marathi", "Kannada"],
        "Semitic": ["Arabic", "Somali", "Maltese"]
    }
    
    # Process by language groups
    for group_name, langs in language_groups.items():
        print(f"\nProcessing {group_name} languages: {', '.join(langs)}")
        logit_lens.analyze_translations(english_sentence, langs)
        # Cleared CUDA cache between language groups due to memory constraints
        if torch.cuda.is_available():
            torch.cuda.empty_cache()