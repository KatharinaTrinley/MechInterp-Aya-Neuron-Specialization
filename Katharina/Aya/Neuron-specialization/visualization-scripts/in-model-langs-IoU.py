import os
import pickle
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob
import matplotlib.patches as mpatches
import copy

#source: https://github.com/Smu-Tan/Neuron-Specialization/tree/main

def load_activations(base_path, layer='layer_0'):
    """Loads layer_0 activations from all .pkl files in subfolders matching 'en-xx'."""
    activations = {}
    subfolders = [f.path for f in os.scandir(base_path) if f.is_dir() and os.path.basename(f.path).startswith('en-')]
    
    for subfolder in subfolders:
        lang_dir = os.path.basename(subfolder)
        pkl_files = glob(os.path.join(subfolder, "*.pkl"))
        
        for pkl_file in pkl_files:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
                if 'layer_0' in data:
                    activations[lang_dir] = data[layer].cpu().numpy()
    
    return activations

def get_specialized_neurons(activations, threshold=0.9):
    """Identifies specialized neurons per language direction based on cumulative activation threshold."""
    specialized_neurons = {}
    for lang, activation in activations.items():
        sorted_indices = np.argsort(-np.abs(activation))  # Sort by absolute activation magnitude
        cumsum = np.cumsum(np.abs(activation[sorted_indices]))
        total_sum = cumsum[-1]
        selected_neurons = sorted_indices[cumsum / total_sum <= threshold]
        specialized_neurons[lang] = set(selected_neurons)
    
    return specialized_neurons

def compute_iou(specialized_neurons):
    """Computes IoU scores between specialized neuron sets of different language directions."""
    languages = list(specialized_neurons.keys())
    num_langs = len(languages)
    iou_matrix = np.zeros((num_langs, num_langs))
    
    for i in range(num_langs):
        for j in range(num_langs):
            set_i = specialized_neurons[languages[i]]
            set_j = specialized_neurons[languages[j]]
            intersection = len(set_i & set_j)
            union = len(set_i | set_j)
            iou_matrix[i, j] = intersection / union if union > 0 else 0
    
    languages = [lang.replace('en-', '') for lang in list(specialized_neurons.keys())]

    return languages, iou_matrix

language_families = {
        # Germanic
        'de': 'Germanic', 'nl': 'Germanic', 'sv': 'Germanic', 'da': 'Germanic', 
        'af': 'Germanic', 'lb': 'Germanic',
        
        # Romance
        'fr': 'Romance', 'es': 'Romance', 'it': 'Romance', 'pt': 'Romance', 
        'ro': 'Romance', 'oc': 'Romance',
        
        # Slavic
        'ru': 'Slavic', 'bg': 'Slavic', 'cs': 'Slavic', 'pl': 'Slavic', 
        'uk': 'Slavic', 'sr': 'Slavic',
        
        # Indo-Aryan
        'hi': 'Indo-Aryan', 'bn': 'Indo-Aryan', 'kn': 'Indo-Aryan', 
        'mr': 'Indo-Aryan', 'gu': 'Indo-Aryan', 'sd': 'Indo-Aryan',
        
        # Afro-Asiatic
        'ar': 'Afro-Asiatic', 'he': 'Afro-Asiatic', 'mt': 'Afro-Asiatic', 
        'am': 'Afro-Asiatic', 'ha': 'Afro-Asiatic', 'ti': 'Afro-Asiatic',
    }

def plot_iou_heatmap(languages, iou_matrix, layer):
    """Plots a heatmap of the IoU scores."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(iou_matrix, xticklabels=languages, yticklabels=languages, cmap='Blues')
    plt.xlabel("Language Direction")
    plt.ylabel("Language Direction")
    plt.title(f"IoU of Specialized Neurons - {layer}")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'IoU_{layer}.png', dpi=300, bbox_inches='tight')
    plt.show()

def reorder_languages(languages, language_families):
    """Reorders the languages based on their language family."""
    # First, sort the languages according to their family order
    ordered_languages = []
    
    # Define the family order
    family_order = ['Germanic', 'Romance', 'Slavic', 'Indo-Aryan', 'Afro-Asiatic']
    
    # Create a mapping from family to languages
    family_to_languages = {family: [] for family in family_order}
    
    for lang in languages:
        family = language_families.get(lang, None)
        if family:
            family_to_languages[family].append(lang)
    
    # Concatenate the languages according to the family order
    for family in family_order:
        ordered_languages.extend(family_to_languages[family])
    
    return ordered_languages

def reorder_iou_matrix(iou_matrix, ordered_languages):
    """Reorders the IoU matrix to match the new language order."""
    ordered_iou_matrix = np.zeros_like(iou_matrix)
    lang_to_index = {lang: i for i, lang in enumerate(ordered_languages)}
    
    # Reorder the IoU matrix rows and columns
    for i, lang_i in enumerate(ordered_languages):
        for j, lang_j in enumerate(ordered_languages):
            ordered_iou_matrix[i, j] = iou_matrix[lang_to_index[lang_i], lang_to_index[lang_j]]
    
    return ordered_iou_matrix

import matplotlib.patches as mpatches

# List of languages to filter
filter_langs = [
    'ar', 'zh', 'cs', 'nl', 'en', 'fr', 'de', 'el', 'he', 'hi', 'id', 'it', 
    'ja', 'ko', 'fa', 'pl', 'pt', 'ro', 'ru', 'es', 'tr', 'uk', 'vi'
]

def filter_languages(languages, iou_matrix, filter_langs):
    """Filters the languages and IoU matrix based on the given list of languages."""
    # Find the intersection of available languages and the filter list
    filtered_languages = [lang for lang in languages if lang in filter_langs]
    # Create a mapping of language to index
    lang_to_index = {lang: idx for idx, lang in enumerate(languages)}
    filtered_indices = [lang_to_index[lang] for lang in filtered_languages]
    
    # Filter the IoU matrix
    filtered_iou_matrix = iou_matrix[np.ix_(filtered_indices, filtered_indices)]
    
    return filtered_languages, filtered_iou_matrix

def plot_iou_heatmap(languages, iou_matrix, layer):
    """Plots a heatmap of the IoU scores with color-coded language families."""
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # Process values for better visualization (similar to first code)
    df = iou_matrix * 100  # Convert to percentage
    df = df.astype(int)
    df_ori = copy.deepcopy(df)
    df = df ** 1.25
    df = (df-df.min())/(df.max()-df.min()) if df.max() != df.min() else df
    df = df * 100
    
    # Create heatmap with darker colors
    sns.heatmap(df, cmap='Blues', ax=ax, xticklabels=languages, yticklabels=languages, 
                linewidths=1, vmin=0, vmax=50)  # Adjusting vmax to make colors darker
    
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_title(layer, y=-0.08, pad=-14)
    
    # Set tick labels with proper rotation
    ax.set_xticklabels(languages, rotation=30)
    ax.set_yticklabels(languages, rotation=0)

    ax.set_xlim(-1, len(languages))
    ax.set_ylim(len(languages), -1)
    
    # Define language family colors (from first code)
    germanic_color = '#93c47d'
    romance_color = '#46bdc6'
    slavic_color = '#8e7cc3'
    aryan_color = '#c27ba0'
    afro_asiatic_color = '#ffd966'
    
    # Create family indices
    family_indices = {
        'Germanic': [],
        'Romance': [],
        'Slavic': [],
        'Indo-Aryan': [],
        'Afro-Asiatic': []
    }
    
    # Populate indices based on language family
    for i, lang in enumerate(languages):
        if lang in language_families:
            family = language_families[lang]
            family_indices[family].append(i)
    
        # Replace the code that adds colored rectangles with this:

    # Germanic
    for i in family_indices['Germanic']:
        # Add rectangles at -1 position (before the first label)
        ax.add_patch(plt.Rectangle((i, -1), 1, 1, fill=True, color=germanic_color))
        ax.add_patch(plt.Rectangle((-1, i), 1, 1, fill=True, color=germanic_color))
        ax.add_patch(plt.Rectangle((i, -1), 1, 1, fill=False, color='white'))
        ax.add_patch(plt.Rectangle((-1, i), 1, 1, fill=False, color='white'))

    # Romance
    for i in family_indices['Romance']:
        ax.add_patch(plt.Rectangle((i, -1), 1, 1, fill=True, color=romance_color))
        ax.add_patch(plt.Rectangle((-1, i), 1, 1, fill=True, color=romance_color))
        ax.add_patch(plt.Rectangle((i, -1), 1, 1, fill=False, color='white'))
        ax.add_patch(plt.Rectangle((-1, i), 1, 1, fill=False, color='white'))

    # Slavic
    for i in family_indices['Slavic']:
        ax.add_patch(plt.Rectangle((i, -1), 1, 1, fill=True, color=slavic_color))
        ax.add_patch(plt.Rectangle((-1, i), 1, 1, fill=True, color=slavic_color))
        ax.add_patch(plt.Rectangle((i, -1), 1, 1, fill=False, color='white'))
        ax.add_patch(plt.Rectangle((-1, i), 1, 1, fill=False, color='white'))

    # Indo-Aryan
    for i in family_indices['Indo-Aryan']:
        ax.add_patch(plt.Rectangle((i, -1), 1, 1, fill=True, color=aryan_color))
        ax.add_patch(plt.Rectangle((-1, i), 1, 1, fill=True, color=aryan_color))
        ax.add_patch(plt.Rectangle((i, -1), 1, 1, fill=False, color='white'))
        ax.add_patch(plt.Rectangle((-1, i), 1, 1, fill=False, color='white'))

    # Afro-Asiatic
    for i in family_indices['Afro-Asiatic']:
        ax.add_patch(plt.Rectangle((i, -1), 1, 1, fill=True, color=afro_asiatic_color))
        ax.add_patch(plt.Rectangle((-1, i), 1, 1, fill=True, color=afro_asiatic_color))
        ax.add_patch(plt.Rectangle((i, -1), 1, 1, fill=False, edgecolor='white'))
        ax.add_patch(plt.Rectangle((-1, i), 1, 1, fill=False, edgecolor='white'))
    
    # Add legend
    legend_labels = ['Germanic', 'Romance', 'Slavic', 'Indo-Aryan', 'Afro-Asiatic']
    legend_colors = [germanic_color, romance_color, slavic_color, aryan_color, afro_asiatic_color]
    legend_handles = [mpatches.Patch(color=legend_colors[i], label=legend_labels[i]) for i in range(len(legend_labels))]
    ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=5, fancybox=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig(f'IoU_{layer}.png', dpi=300, bbox_inches='tight')
    plt.show()


base_path = "/netscratch/ktrinley/MechInterp-Project/Aya/activations_aya-23-8B"
for layer_index in range(32):
    layer_name = f'layer_{layer_index}'  
    
    activations = load_activations(base_path, layer_name)
    
    specialized_neurons = get_specialized_neurons(activations)
    
    languages, iou_matrix = compute_iou(specialized_neurons)

    filtered_languages, filtered_iou_matrix = filter_languages(languages, iou_matrix, filter_langs)

    ordered_iou_matrix = reorder_iou_matrix(filtered_iou_matrix, filtered_languages)

    ordered_languages = reorder_languages(filtered_languages, language_families)


    plot_iou_heatmap(ordered_languages, ordered_iou_matrix, layer_name)