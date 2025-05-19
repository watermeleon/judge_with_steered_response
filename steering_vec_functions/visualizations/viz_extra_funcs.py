from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def process_category_statistics(responses, cat_param= 'category_id', metric_name='metric_score'):
    """
    Process responses to extract mean and std per category for different judge types.
    
    Parameters:
    responses (list): List of response dictionaries containing judge results
    
    Returns:
    dict: Nested dictionary with statistics per category and judge type
    """

    # Initialize data storage
    category_data = defaultdict(lambda: {
        'single_base': [],
        'single_suggestive': [],
        'paired_a': [],
        'paired_b': [],
        'steered_base': [],
        'steered_suggestive': []
    })
    
    # Process each response
    for response in responses:
        category = response['full_data'][cat_param]
        
        # Extract scores from different judge types
        if 'judge_single' in response:
            category_data[category]['single_base'].append(
                response['judge_single']['base'][metric_name])
            category_data[category]['single_suggestive'].append(
                response['judge_single']['suggestive'][metric_name])
        
        if 'judge_base_vs_suggestive' in response:
            category_data[category]['paired_a'].append(
                response['judge_base_vs_suggestive']['response_A'][metric_name])
            category_data[category]['paired_b'].append(
                response['judge_base_vs_suggestive']['response_B'][metric_name])
        
        if 'judge_base_steered_pair' in response:
            category_data[category]['steered_base'].append(
                response['judge_base_steered_pair']['response_A'][metric_name])
        
        if 'judge_suggestive_steered_pair' in response:
            category_data[category]['steered_suggestive'].append(
                response['judge_suggestive_steered_pair']['response_A'][metric_name])
    
    # Calculate statistics
    statistics = {}
    for category, scores in category_data.items():
        statistics[category] = {}
        for judge_type, score_list in scores.items():
            if score_list:
                statistics[category][judge_type] = {
                    'mean': np.mean(score_list),
                    'std': np.std(score_list),
                    'count': len(score_list)
                }
    
    return statistics


def visualize_category_statistics(statistics, figsize=(14, 8), skip_paired=True):
    """
    Create visualization showing mean scores per category for different judge types.
    
    Parameters:
    statistics (dict): Statistics dictionary from process_category_statistics
    figsize (tuple): Figure size
    
    Returns:
    matplotlib.figure.Figure: The generated figure
    """
    # Prepare data for plotting
    categories = sorted(statistics.keys())
    
    # Define judge groups for visualization
    judge_groups = {
        'Single Evaluation': ['single_base', 'single_suggestive'],
        'Paired Evaluation': ['paired_a', 'paired_b'],
        'Steered Evaluation': ['steered_base', 'steered_suggestive']
    }
    if skip_paired:
        judge_groups.pop('Paired Evaluation')
    
    # Create subplots
    num_plots = len(judge_groups)
    fig, axes = plt.subplots(1, num_plots, figsize=figsize)
    fig.suptitle('Mean Scores by Category and Judge Type', fontsize=16, y=1.02)
    
    colors = {'base': '#3274A1', 'suggestive': '#E1812C', 'a': '#3274A1', 'b': '#E1812C'}
    
    for idx, (group_name, judge_types) in enumerate(judge_groups.items()):
        ax = axes[idx]
        bar_width = 0.35
        x_positions = np.arange(len(categories))
        
        for i, judge_type in enumerate(judge_types):
            means = []
            stds = []
            
            for category in categories:
                if category in statistics and judge_type in statistics[category]:
                    means.append(statistics[category][judge_type]['mean'])
                    stds.append(statistics[category][judge_type]['std'])
                else:
                    means.append(0)
                    stds.append(0)
            
            # Determine color and label
            if 'base' in judge_type or '_a' in judge_type:
                color = colors['base']
                label = 'Base'
            else:
                color = colors['suggestive']
                label = 'Suggestive'
            # Create bars
            x_offset = -bar_width/2 if i == 0 else bar_width/2
            ax.bar(x_positions + x_offset, means, bar_width, 
                  yerr=stds, capsize=5, color=color, alpha=0.8, label=label)
        
        ax.set_title(group_name, fontsize=14)
        ax.set_ylabel('Score' if idx == 0 else '', fontsize=12)
        ax.set_xticks(x_positions)
        # ax.set_xticklabels(categories, rotation=45, ha='center', fontsize=10)
        ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.set_ylim(0, 10)  # Set y-axis maximum to 10
    
    plt.tight_layout()
    return fig


def create_category_summary_table(statistics):
    """
    Create a summary table showing statistics for all categories.
    
    Parameters:
    statistics (dict): Statistics dictionary from process_category_statistics
    
    Returns:
    pandas.DataFrame: Summary table
    """
    data = []
    
    for category in sorted(statistics.keys()):
        row = {'Category': category}
        
        # Add mean values for key comparisons
        stat = statistics[category]
        
        row['Single_Base'] = f"{stat.get('single_base', {}).get('mean', 0):.2f}"
        row['Single_Suggestive'] = f"{stat.get('single_suggestive', {}).get('mean', 0):.2f}"
        row['Paired_A'] = f"{stat.get('paired_a', {}).get('mean', 0):.2f}"
        row['Paired_B'] = f"{stat.get('paired_b', {}).get('mean', 0):.2f}"
        
        data.append(row)
    
    return pd.DataFrame(data)

import numpy as np
from sentence_transformers import SentenceTransformer
import time


def analyze_text_variations(base_responses, suggestive_responses, base_steered_responses, suggestive_steered_responses):
    # Load pre-trained model for text embeddings
    # model = SentenceTransformer('all-MiniLM-L6-v2')
    # load answerdotai/ModernBERT-base
    model = SentenceTransformer('answerdotai/ModernBERT-base')
    # load roberta-base
    # model = SentenceTransformer('roberta-base')
    # model = SentenceTransformer('sentence-transformers/all-roberta-large-v1')
    start_time = time.time()
    print("Encoding responses...")
    subset = -1
    base_encodings = model.encode(base_responses)
    print("Base responses encoded in", time.time() - start_time, "seconds")
    suggestive_encodings = model.encode(suggestive_responses)
    print("Finished encoding suggestive responses in", time.time() - start_time, "seconds")
    base_steered_encodings = model.encode(base_steered_responses)
    print("Finished encoding base steered responses in", time.time() - start_time, "seconds")
    suggestive_steered_encodings = model.encode(suggestive_steered_responses)
    print("Finished encoding suggestive steered responses in", time.time() - start_time, "seconds")

    # Combine all responses into a dictionary
    response_classes = {
        "Base": base_encodings,
        "Provoked": suggestive_encodings,
        "Base Steered": base_steered_encodings,
        "Provoked Steered": suggestive_steered_encodings
    }

    # Function to calculate cosine similarity between two vectors
    def cosine_similarity(vec1, vec2):
        # vec1 = np.asarray(vec1, dtype=np.float32)
        # vec2 = np.asarray(vec2, dtype=np.float32)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    # 1. Calculate variation within each response class
    variation_results = {}
    from tqdm import tqdm
    for name, embeddings in tqdm(response_classes.items()):
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                similarities.append(cosine_similarity(embeddings[i], embeddings[j]))
        
        # Calculate statistics
        similarities = np.array(similarities)
        variation_results[name] = {
            'mean_similarity': np.mean(similarities),
            'variation_score': 1 - np.mean(similarities)  # Higher value = more variation
        }
    
    # 2. Calculate similarity between response classes for each index
    class_names = list(response_classes.keys())
    pair_similarities = {}
    
    # Initialize structures for storing similarities
    for i in range(len(class_names)):
        for j in range(i+1, len(class_names)):
            pair = f"{class_names[i]} vs {class_names[j]}"
            pair_similarities[pair] = []
    
    # Calculate similarities for each index
    for idx in tqdm(range(len(base_responses))):
        for i in range(len(class_names)):
            for j in range(i+1, len(class_names)):
                name1, name2 = class_names[i], class_names[j]
                pair = f"{name1} vs {name2}"
                
                emb1 = response_classes[name1][idx]
                emb2 = response_classes[name2][idx]
                
                sim = cosine_similarity(emb1, emb2)
                pair_similarities[pair].append(sim)
    
    # Calculate mean similarities for each pair
    mean_pair_similarities = {pair: np.mean(sims) for pair, sims in pair_similarities.items()}
    
    # 3. Calculate mean similarity of each class to all others
    class_mean_similarities = {}
    
    for name in class_names:
        relevant_pairs = [p for p in mean_pair_similarities.keys() if name in p]
        mean_sim = np.mean([mean_pair_similarities[p] for p in relevant_pairs])
        class_mean_similarities[name] = mean_sim
    
    return variation_results, mean_pair_similarities, class_mean_similarities