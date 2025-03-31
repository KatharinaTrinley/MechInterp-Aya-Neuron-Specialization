import os
import pickle
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob
import matplotlib.patches as mpatches
import copy
from matplotlib.colors import LinearSegmentedColormap

#source: https://github.com/Smu-Tan/Neuron-Specialization/tree/main

def load_activations(base_path, layer='layer_0'):
    """Loads layer activations from all .pkl files in subfolders matching 'en-xx'."""
    activations = {}
    subfolders = [f.path for f in os.scandir(base_path) if f.is_dir() and os.path.basename(f.path).startswith('en-')]
    
    for subfolder in subfolders:
        lang_dir = os.path.basename(subfolder)
        pkl_files = glob(os.path.join(subfolder, "*.pkl"))
        
        for pkl_file in pkl_files:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
                if layer in data:
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

# Language families for reference
language_families = {
    # Germanic
    'de': 'Germanic', 'nl': 'Germanic', 'sv': 'Germanic', 'da': 'Germanic',
    'af': 'Germanic', 'lb': 'Germanic', 'no': 'Germanic', 'is': 'Germanic',
    'en': 'Germanic',
    
    # Romance
    'fr': 'Romance', 'es': 'Romance', 'it': 'Romance', 'pt': 'Romance',
    'ro': 'Romance', 'oc': 'Romance', 'ast': 'Romance', 'ca': 'Romance',
    
    # Slavic
    'ru': 'Slavic', 'cs': 'Slavic', 'pl': 'Slavic', 'bg': 'Slavic',
    'uk': 'Slavic', 'sr': 'Slavic', 'be': 'Slavic', 'bs': 'Slavic',
    
    # Indo-Aryan
    'hi': 'Indo-Aryan', 'bn': 'Indo-Aryan', 'kn': 'Indo-Aryan', 'mr': 'Indo-Aryan',
    'sd': 'Indo-Aryan', 'gu': 'Indo-Aryan', 'ne': 'Indo-Aryan', 'ur': 'Indo-Aryan',
    
    # Afro-Asiatic
    'ar': 'Afro-Asiatic', 'he': 'Afro-Asiatic', 'mt': 'Afro-Asiatic',
    'am': 'Afro-Asiatic', 'ti': 'Afro-Asiatic', 'ha': 'Afro-Asiatic',
    'kab': 'Afro-Asiatic', 'so': 'Afro-Asiatic',
}

# EC40 languages with their resourcedness
ec40_resourcedness = {
    # High resource (5M)
    'de': 'High', 'nl': 'High', 'fr': 'High', 'es': 'High', 'ru': 'High',
    'cs': 'High', 'hi': 'High', 'bn': 'High', 'ar': 'High', 'he': 'High',
    
    # Medium resource (1M)
    'sv': 'Medium', 'da': 'Medium', 'it': 'Medium', 'pt': 'Medium', 'pl': 'Medium',
    'bg': 'Medium', 'kn': 'Medium', 'mr': 'Medium', 'mt': 'Medium', 'ha': 'Medium',
    
    # Low resource (100k)
    'af': 'Low', 'lb': 'Low', 'ro': 'Low', 'oc': 'Low', 'uk': 'Low',
    'sr': 'Low', 'sd': 'Low', 'gu': 'Low', 'ti': 'Low', 'am': 'Low',
    
    # Extremely-Low resource (50k)
    'no': 'Extremely-Low', 'is': 'Extremely-Low', 'ast': 'Extremely-Low', 'ca': 'Extremely-Low',
    'be': 'Extremely-Low', 'bs': 'Extremely-Low', 'ne': 'Extremely-Low', 'ur': 'Extremely-Low',
    'kab': 'Extremely-Low', 'so': 'Extremely-Low'
}

# Languages in Aya-23
aya_languages = {
    'ar', 'cs', 'de', 'fr', 'he', 'hi', 'it', 'nl', 'pl', 'pt', 'ro', 'ru', 'es', 'uk'
}

# Hardcoded language order with HRL first, then LRL, grouped by language families
hardcoded_language_order = [
    # Germanic Family - HRL
    'de', 'nl',
    # Germanic Family - LRL
    'sv', 'da', 'af', 'lb', 'no', 'is',
    
    # Romance Family - HRL
    'fr', 'es',
    # Romance Family - LRL
    'it', 'pt', 'ro', 'oc', 'ast', 'ca',
    
    # Slavic Family - HRL
    'ru', 'cs',
    # Slavic Family - LRL
    'pl', 'bg', 'uk', 'sr', 'be', 'bs',
    
    # Indo-Aryan Family - HRL
    'hi', 'bn',
    # Indo-Aryan Family - LRL
    'kn', 'mr', 'sd', 'gu', 'ne', 'ur',
    
    # Afro-Asiatic Family - HRL
    'ar', 'he',
    # Afro-Asiatic Family - LRL
    'mt', 'am', 'ti', 'ha', 'kab', 'so'
]

def reorder_iou_matrix_by_hardcoded_order(languages, iou_matrix):
    """Reorders the IoU matrix to match the hardcoded language order."""
    # Find which languages from the hardcoded order are available in the data
    available_languages = []
    for lang in hardcoded_language_order:
        if lang in languages:
            available_languages.append(lang)
    
    # If no languages match, return the original data
    if not available_languages:
        return languages, iou_matrix
    
    # Create a mapping of language to index in the original matrix
    lang_to_index = {lang: idx for idx, lang in enumerate(languages)}
    
    # Get indices of available languages in the original matrix
    indices = [lang_to_index[lang] for lang in available_languages if lang in lang_to_index]
    
    # Reorder the matrix
    reordered_iou_matrix = iou_matrix[np.ix_(indices, indices)]
    
    return available_languages, reordered_iou_matrix

def plot_iou_heatmap(languages, iou_matrix, layer):
    """Plots a heatmap of the IoU scores with color-coded language families and resourcedness."""
    fig, ax = plt.subplots(figsize=(18, 14))
    
    # Process values for better visualization
    df = iou_matrix * 100  # Convert to percentage
    df = df.astype(int)
    df_ori = copy.deepcopy(df)
    df = df ** 1.25
    df = (df-df.min())/(df.max()-df.min()) if df.max() != df.min() else df
    df = df * 100
    
    # Create heatmap with darker colors
    sns.heatmap(df, cmap='Blues', ax=ax, xticklabels=languages, yticklabels=languages, 
                linewidths=1, vmin=0, vmax=50)
    
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_title(f"{layer} - Language Neuron Specialization IoU (EC40)", fontsize=18, pad=20)
    
    # Set tick labels with proper rotation and color based on resourcedness
    x_tick_labels = ax.get_xticklabels()
    y_tick_labels = ax.get_yticklabels()
    
    for label in x_tick_labels:
        lang = label.get_text()
        # Set color based on presence in Aya and resourcedness
        if lang in aya_languages:
            # All languages in Aya are marked as high-resource (green)
            label.set_color('green')
            label.set_fontweight('bold')
        elif lang in ec40_resourcedness:
            resourcedness = ec40_resourcedness[lang]
            if resourcedness == 'High':
                label.set_color('green')
                label.set_fontweight('bold')
            elif resourcedness == 'Medium':
                label.set_color('orange')
            elif resourcedness == 'Low':
                label.set_color('red')
            elif resourcedness == 'Extremely-Low':
                label.set_color('red')
                label.set_alpha(0.7)  # Slightly more transparent for Extremely-Low
    
    for label in y_tick_labels:
        lang = label.get_text()
        # Set color based on presence in Aya and resourcedness
        if lang in aya_languages:
            # All languages in Aya are marked as high-resource (green)
            label.set_color('green')
            label.set_fontweight('bold')
        elif lang in ec40_resourcedness:
            resourcedness = ec40_resourcedness[lang]
            if resourcedness == 'High':
                label.set_color('green')
                label.set_fontweight('bold')
            elif resourcedness == 'Medium':
                label.set_color('orange')
            elif resourcedness == 'Low':
                label.set_color('red')
            elif resourcedness == 'Extremely-Low':
                label.set_color('red')
                label.set_alpha(0.7)  # Slightly more transparent for Extremely-Low
    
    ax.set_xticklabels(x_tick_labels, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(y_tick_labels, rotation=0, fontsize=10)

    ax.set_xlim(-1, len(languages))
    ax.set_ylim(len(languages), -1)
    
    # Define language family colors
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
    
    # Add colored rectangles for language families
    # Germanic
    for i in family_indices['Germanic']:
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
    
    # family legend
    family_legend_labels = ['Germanic', 'Romance', 'Slavic', 'Indo-Aryan', 'Afro-Asiatic']
    family_legend_colors = [germanic_color, romance_color, slavic_color, aryan_color, afro_asiatic_color]
    family_legend_handles = [mpatches.Patch(color=family_legend_colors[i], label=family_legend_labels[i]) 
                            for i in range(len(family_legend_labels))]
    
    # resourcedness legend
    resourcedness_legend_labels = [
        'High resource (HRL)', 
        'Medium resource (LRL)', 
        'Low resource (LRL)',
        'Extremely-Low resource (LRL)'
    ]
    resourcedness_legend_colors = ['green', 'orange', 'red', 'red']
    
    resourcedness_legend_handles = []
    for i, (color, label) in enumerate(zip(resourcedness_legend_colors, resourcedness_legend_labels)):
        if i == 3:  # Extremely-Low
            handle = plt.Line2D([0], [0], color=color, marker='o', linestyle='', markersize=10, alpha=0.7, label=label)
        else:
            handle = plt.Line2D([0], [0], color=color, marker='o', linestyle='', markersize=10, label=label)
        resourcedness_legend_handles.append(handle)
    
    # 2 legends
    family_legend = ax.legend(handles=family_legend_handles, loc='upper left', 
                              bbox_to_anchor=(0.0, 1.15), ncol=5, 
                              title="Language Families", fontsize=10, title_fontsize=12)
    
    ax.add_artist(family_legend)
    
    ax.legend(handles=resourcedness_legend_handles, loc='upper right', 
              bbox_to_anchor=(1.0, 1.15), ncol=2,
              title="EC40 Resourcedness Levels", fontsize=10, title_fontsize=12)
    
    description = (
        "Visualization shows the Intersection over Union (IoU) of specialized neurons across languages.\n"
        "Languages are ordered by families with HRL (High-Resource Languages) followed by LRL (Low-Resource Languages).\n"
        "Green: High-Resource Languages (5M tokens) | Orange: Medium-Resource (1M) | Red: Low/Extremely-Low Resource (≤100k)"
    )
    
    plt.figtext(0.5, 0.01, description, ha='center', fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.5'))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(f'IoU_{layer}_EC40_HRL_LRL.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory

base_path = "/netscratch/ktrinley/MechInterp-Project/Aya/activations_aya-23-8B"

for layer_index in range(32):
    layer_name = f'layer_{layer_index}'  
    print(f"Processing {layer_name}...")
    
    # Load activations for the current layer
    activations = load_activations(base_path, layer_name)
    
    # Get specialized neurons for the current layer
    specialized_neurons = get_specialized_neurons(activations)
    
    # Compute IoU matrix for the current layer
    languages, iou_matrix = compute_iou(specialized_neurons)

    print(f"Found {len(languages)} languages in the data")
    
    # Use the hardcoded order to reorder the language matrix
    ordered_languages, ordered_iou_matrix = reorder_iou_matrix_by_hardcoded_order(languages, iou_matrix)

    # Plot the heatmap for the current layer
    plot_iou_heatmap(ordered_languages, ordered_iou_matrix, layer_name)
    print(f"Completed {layer_name}")