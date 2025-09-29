from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def process_category_statistics(responses, cat_param= 'category_id', metric_name='metric_score', return_cat_data=False):
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
    
    if return_cat_data is True:
        return statistics, category_data
    else:
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


def calculate_category_auroc(responses, cat_param='category_id', metric_name='metric_score', judge_type='single', leave_out_cats=None):
    """
    Calculate AU-ROC scores between base and suggestive responses for each category.
    
    Parameters:
    responses (list): List of response dictionaries containing judge results
    cat_param (str): Parameter name for category identification (default: 'category_id')
    metric_name (str): Metric to calculate AU-ROC for (default: 'metric_score')
    judge_type (str): Type of judge evaluation - 'single' or 'steered' (default: 'single')
    
    Returns:
    pandas.DataFrame: DataFrame with categories and their AU-ROC scores
    """
    # Get category statistics using the existing function
    _, category_data = process_category_statistics(
        responses, cat_param=cat_param, metric_name=metric_name, return_cat_data=True
    )
    
    auroc_results = []

    for category in sorted(category_data.keys()):
        if leave_out_cats and category in leave_out_cats:
            continue
        cat_data = category_data[category]
        
        # Get base and suggestive scores based on judge type
        if judge_type == 'single':
            base_scores = cat_data.get('single_base', [])
            suggestive_scores = cat_data.get('single_suggestive', [])
        elif judge_type == 'steered':
            base_scores = cat_data.get('steered_base', [])
            suggestive_scores = cat_data.get('steered_suggestive', [])
        else:
            raise ValueError("judge_type must be 'single' or 'steered'")
        
        if len(base_scores) > 0 and len(suggestive_scores) > 0:
            # Ensure we have the same number of samples for fair comparison
            min_samples = min(len(base_scores), len(suggestive_scores))
            base_scores = base_scores[:min_samples]
            suggestive_scores = suggestive_scores[:min_samples]
            
            # Create labels (0 for base, 1 for suggestive)
            y_true = [0] * len(base_scores) + [1] * len(suggestive_scores)

            # Combine scores
            y_scores = base_scores + suggestive_scores
            
            # Calculate AU-ROC only if we have both classes
            if len(set(y_true)) > 1:
                try:
                    auroc = roc_auc_score(y_true, y_scores)
                except ValueError:
                    # Handle case where all scores are the same
                    auroc = 0.5
            else:
                auroc = 0.5  # Default when only one class present
                
            auroc_results.append({
                'Category': category,
                'AU_ROC': auroc,
                'Base_Mean': np.mean(base_scores),
                'Suggestive_Mean': np.mean(suggestive_scores),
                'Base_Count': len(base_scores),
                'Suggestive_Count': len(suggestive_scores),
                'Metric': metric_name,
                'Judge_Type': judge_type.capitalize()
            })
    
    # Create DataFrame
    df = pd.DataFrame(auroc_results)
    
    # Sort by AU-ROC score for easier interpretation
    if not df.empty:
        df = df.sort_values('AU_ROC', ascending=False).reset_index(drop=True)
        
        # Add interpretation column
        df['Interpretation'] = df['AU_ROC'].apply(
            lambda x: 'Perfect Separation' if x == 1.0 
                     else 'Strong Separation' if x > 0.8
                     else 'Good Separation' if x > 0.7
                     else 'Fair Separation' if x > 0.6
                     else 'Poor Separation' if x > 0.5
                     else 'No Separation' if x == 0.5
                     else 'Reverse Separation'
        )
    
    return df


def calculate_multiple_metrics_auroc(responses, cat_param='category_id', metrics=['correctness', 'metric_score']):
    """
    Calculate AU-ROC scores for multiple metrics across categories.
    
    Parameters:
    responses (list): List of response dictionaries containing judge results
    cat_param (str): Parameter name for category identification (default: 'category_id')
    metrics (list): List of metric names to calculate AU-ROC for
    
    Returns:
    pandas.DataFrame: DataFrame with categories, metrics, and their AU-ROC scores
    """
    all_results = []
    
    for metric in metrics:
        df = calculate_category_auroc(responses, cat_param=cat_param, metric_name=metric)
        all_results.append(df)
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Pivot table for easier comparison
    pivot_df = combined_df.pivot_table(
        index='Category', 
        columns='Metric', 
        values='AU_ROC', 
        fill_value=np.nan
    ).reset_index()
    
    return combined_df, pivot_df


def calculate_judge_comparison_auroc(responses, cat_param='category_id', metrics=['correctness', 'metric_score'], leave_out_cats=None):
    """
    Calculate AU-ROC scores for both Single and Steered judge types across categories and metrics.
    
    Parameters:
    responses (list): List of response dictionaries containing judge results
    cat_param (str): Parameter name for category identification (default: 'category_id')
    metrics (list): List of metric names to calculate AU-ROC for
    
    Returns:
    tuple: (single_results_df, steered_results_df, comparison_pivot_df)
        - single_results_df: DataFrame with Single judge AU-ROC results
        - steered_results_df: DataFrame with Steered judge AU-ROC results  
        - comparison_pivot_df: Pivot table comparing both judge types
    """
    single_results = []
    steered_results = []
    
    for metric in metrics:
        # Calculate for Single judge
        single_df = calculate_category_auroc(responses, cat_param=cat_param, metric_name=metric, judge_type='single', leave_out_cats=leave_out_cats)
        single_results.append(single_df)
        
        # Calculate for Steered judge
        steered_df = calculate_category_auroc(responses, cat_param=cat_param, metric_name=metric, judge_type='steered', leave_out_cats=leave_out_cats)
        steered_results.append(steered_df)
    
    # Combine results
    single_combined = pd.concat(single_results, ignore_index=True) if single_results else pd.DataFrame()
    steered_combined = pd.concat(steered_results, ignore_index=True) if steered_results else pd.DataFrame()
    
    # Create comparison pivot table
    all_results = pd.concat([single_combined, steered_combined], ignore_index=True)
    
    if not all_results.empty:
        comparison_pivot = all_results.pivot_table(
            index=['Category', 'Metric'], 
            columns='Judge_Type', 
            values='AU_ROC', 
            fill_value=np.nan
        ).reset_index()
        
        # Flatten column names
        comparison_pivot.columns.name = None
        
        # Calculate difference between Single and Steered
        if 'Single' in comparison_pivot.columns and 'Steered' in comparison_pivot.columns:
            comparison_pivot['Difference (Single - Steered)'] = (
                comparison_pivot['Single'] - comparison_pivot['Steered']
            )
    else:
        comparison_pivot = pd.DataFrame()
    
    return single_combined, steered_combined, comparison_pivot

import numpy as np
from sentence_transformers import SentenceTransformer
import time
from sklearn.metrics import roc_auc_score


def analyze_text_variations(base_responses, suggestive_responses, base_steered_responses, suggestive_steered_responses):
    # Load pre-trained model for text embeddings
    model = SentenceTransformer('answerdotai/ModernBERT-base')
    start_time = time.time()

    print("Encoding responses...")
    subset = -1
    base_encodings = model.encode(base_responses)
    print(f"Base responses encoded in {time.time() - start_time:.2f} seconds")
    suggestive_encodings = model.encode(suggestive_responses)
    print(f"Suggestive responses encoded in {time.time() - start_time:.2f} seconds")
    base_steered_encodings = model.encode(base_steered_responses)
    print(f"Base steered responses encoded in {time.time() - start_time:.2f} seconds")
    suggestive_steered_encodings = model.encode(suggestive_steered_responses)
    print(f"Suggestive steered responses encoded in {time.time() - start_time:.2f} seconds")

    # Combine all responses into a dictionary
    response_classes = {
        "Base": base_encodings,
        "Provoked": suggestive_encodings,
        "Base Steered": base_steered_encodings,
        "Provoked Steered": suggestive_steered_encodings
    }

    # Function to calculate cosine similarity between two vectors
    def cosine_similarity(vec1, vec2):
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


import matplotlib.pyplot as plt
from notebooks.notebook_functions import get_judge_result_file




def plot_combined_llm_comparison_Manip(comparison_pivot_llm1, comparison_pivot_llm2, 
                                llm1_name="Claude", llm2_name="GPT4",
                                metric='metric_score', figsize=(16, 8), 
                                title_prefix="Judge Performance Comparison", 
                                save_path=None, show_values=True, font_size_multiplier=1.0,
                                leave_out_cats=None, colors_bars=None):
    """
    Plot comparison between Single judge and JUSSA (steered judge) performance
    for two different LLMs combined in one plot.
    
    Parameters:
    -----------
    comparison_pivot_llm1 : pd.DataFrame
        DataFrame for first LLM with columns: Category, Metric, Single, Steered, Difference
    comparison_pivot_llm2 : pd.DataFrame
        DataFrame for second LLM with columns: Category, Metric, Single, Steered, Difference
    llm1_name : str
        Name of the first LLM for labeling
    llm2_name : str
        Name of the second LLM for labeling
    metric : str
        Which metric to plot ('metric_score' or 'correctness')
    figsize : tuple
        Figure size (width, height)
    title_prefix : str
        Prefix for the plot title
    save_path : str, optional
        Path to save the figure
    show_values : bool
        Whether to show values on top of bars
    font_size_multiplier : float
        Factor to scale all font sizes
    leave_out_cats : list, optional
        Categories to exclude from the plot
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Rectangle

    if colors_bars is None:
        # Updated color scheme for 4 bars (2 LLMs x 2 judge types)
        colors_bars = {
            'claude_single': "#003f5c",      # Dark blue
            'claude_steered': "#374c80",     # Medium blue  
            'gpt4_single': "#bc5090",        # Pink/purple
            'gpt4_steered': "#ff6361",       # Red/orange
        }

    
    group_colors = {
        'Skewed Presentation': '#E6F2FF',
        'Misleading Claims': '#FFF2E6',
        'Emotional Pressure': '#F2FFE6'
    }

    # Define category grouping
    category_groups = {
        # Group 1: Skewed Presentation
        'cherry_picked_evidence': 'Skewed Presentation',
        'false_dichotomy': 'Skewed Presentation',
        'false_transparency': 'Skewed Presentation',
        'assistant_self_interest': 'Skewed Presentation',
        'strategic_omission': 'Skewed Presentation',
        
        # Group 2: Misleading Claims and False Information
        'risk_distortion': 'Misleading Claims',
        'false_causality': 'Misleading Claims',
        'manufactured_consensus': 'Misleading Claims',
        'false_credentials': 'Misleading Claims',
        'hidden_premises': 'Misleading Claims',
        
        # Group 3: Emotional and Psychological Pressure
        'emotional_manipulation': 'Emotional Pressure',
        'manufactured_scarcity': 'Emotional Pressure',
        'artificial_time_pressure': 'Emotional Pressure'
    }

    category_display_names = {
        'cherry_picked_evidence': 'Cherry Picked\n Evidence',
        'false_dichotomy': 'False\n Dichotomy', 
        'false_transparency': 'False\n Transparency',
        'assistant_self_interest': 'Assistant\n Self Interest',
        'strategic_omission': 'Strategic\n Omission',
        'risk_distortion': 'Risk\n Distortion',
        'false_causality': 'False\n Causality',
        'manufactured_consensus': 'Manufactured\n Consensus',
        'false_credentials': 'False\n Credentials',
        'hidden_premises': 'Hidden\n Premises',
        'emotional_manipulation': 'Emotional\n Manipulation',
        'manufactured_scarcity': 'Manufactured\n Scarcity',
        'artificial_time_pressure': 'Artificial Time\n Pressure'
    }   
    
    # Filter data for the specified metric
    metric_data_llm1 = comparison_pivot_llm1[comparison_pivot_llm1['Metric'] == metric].copy()
    metric_data_llm2 = comparison_pivot_llm2[comparison_pivot_llm2['Metric'] == metric].copy()
    
    if metric_data_llm1.empty or metric_data_llm2.empty:
        raise ValueError(f"No data found for metric: {metric}")
    
    # Apply leave_out_cats filter
    if leave_out_cats is not None:
        metric_data_llm1 = metric_data_llm1[~metric_data_llm1['Category'].isin(leave_out_cats)]
        metric_data_llm2 = metric_data_llm2[~metric_data_llm2['Category'].isin(leave_out_cats)]
        category_groups = {cat: group for cat, group in category_groups.items() if cat not in leave_out_cats}
    
    # Get common categories (intersection of both LLMs)
    categories_llm1 = set(metric_data_llm1['Category'].tolist())
    categories_llm2 = set(metric_data_llm2['Category'].tolist())
    available_categories = list(categories_llm1.intersection(categories_llm2))
    
    # Group by higher-level category
    grouped_categories = {}
    for cat in available_categories:
        group = category_groups.get(cat, 'Other')
        if group not in grouped_categories:
            grouped_categories[group] = []
        grouped_categories[group].append(cat)
    
    # Sort categories within each group alphabetically
    for group in grouped_categories:
        grouped_categories[group].sort()
    
    # Define the display order of the groups
    group_order = ['Skewed Presentation', 'Misleading Claims', 'Emotional Pressure', 'Other']
    
    # Create a flat list of categories in the correct order
    categories = []
    for group in group_order:
        if group in grouped_categories:
            categories.extend(grouped_categories[group])
    
    # Filter and reindex data to match our ordered categories
    metric_data_llm1 = metric_data_llm1.set_index('Category').reindex(categories).reset_index()
    metric_data_llm2 = metric_data_llm2.set_index('Category').reindex(categories).reset_index()
    
    # Get values for all 4 bar types
    llm1_single_scores = metric_data_llm1['Single'].values
    llm1_steered_scores = metric_data_llm1['Steered'].values
    llm2_single_scores = metric_data_llm2['Single'].values
    llm2_steered_scores = metric_data_llm2['Steered'].values
    
    # Print mean differences
    print(f"\n=== {llm1_name} Results ===")
    print(f"Mean AUROC (Single): {np.mean(llm1_single_scores):.4f}")
    print(f"Mean AUROC (Steered): {np.mean(llm1_steered_scores):.4f}")
    print(f"Mean AUROC Difference: {np.mean(llm1_steered_scores) - np.mean(llm1_single_scores):.4f}")
    
    print(f"\n=== {llm2_name} Results ===")
    print(f"Mean AUROC (Single): {np.mean(llm2_single_scores):.4f}")
    print(f"Mean AUROC (Steered): {np.mean(llm2_steered_scores):.4f}")
    print(f"Mean AUROC Difference: {np.mean(llm2_steered_scores) - np.mean(llm2_single_scores):.4f}")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set up bar positions with group spacing - now for 4 bars per category
    bar_width = 0.22  # Narrower bars to fit 4
    category_gap = 0.3
    extra_group_gap = 0.4  # Additional gap between different higher-level groups
    
    # Calculate positions for all categories with extra space between groups
    x_positions = []
    current_x = 0
    last_group = None
    
    for i, category in enumerate(categories):
        group = category_groups.get(category, 'Other')
        
        # Add extra gap between different groups
        if last_group is not None and group != last_group:
            current_x += extra_group_gap
        
        x_positions.append(current_x)
        current_x += 4 * bar_width + category_gap  # Space for 4 bars
        last_group = group
    
    x = np.array(x_positions)

    # Extra gap between bars2 and bars3
    gap_dif_models = 0.05  # adjust this value as needed

    # Create bars - 4 bars per category
    bars1 = ax.bar(x - 1.5*bar_width, llm1_single_scores, bar_width, 
                label=f'{llm1_name} Single', color=colors_bars['claude_single'], 
                alpha=0.9, edgecolor='white', linewidth=0.7)

    bars2 = ax.bar(x - 0.5*bar_width, llm1_steered_scores, bar_width, 
                label=f'{llm1_name} JUSSA', color=colors_bars['claude_steered'], 
                alpha=0.9, edgecolor='white', linewidth=0.7)

    bars3 = ax.bar(x + 0.5*bar_width + gap_dif_models, llm2_single_scores, bar_width, 
                label=f'{llm2_name} Single', color=colors_bars['gpt4_single'], 
                alpha=0.9, edgecolor='white', linewidth=0.7)

    bars4 = ax.bar(x + 1.5*bar_width + gap_dif_models, llm2_steered_scores, bar_width, 
                label=f'{llm2_name} JUSSA', color=colors_bars['gpt4_steered'], 
                alpha=0.9, edgecolor='white', linewidth=0.7)
    
    # Add background coloring for each group
    last_group = None
    group_start_x = 0
    group_spans = {}  # Store the x-spans of each group
    
    for i, category in enumerate(categories):
        group = category_groups.get(category, 'Other')
        
        # Check if we're starting a new group
        if last_group != group:
            if last_group is not None:
                # Save the span of the previous group
                end_x = x[i-1] + 1.5*bar_width + bar_width/2
                group_spans[last_group] = (group_start_x, end_x)
            
            # Start of a new group
            group_start_x = x[i] - 1.5*bar_width - bar_width/2
            last_group = group
    
    # Add the last group
    if last_group is not None:
        end_x = x[-1] + 1.5*bar_width + bar_width/2
        group_spans[last_group] = (group_start_x, end_x)
    
    # Draw background rectangles for groups
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max * 1.1)  # Make room for group labels
    
    for group, (start_x, end_x) in group_spans.items():
        width = end_x - start_x
        rect = Rectangle((start_x, y_min), width, y_max - y_min,
                         color=group_colors.get(group, '#FFFFFF'),
                         alpha=0.3, zorder=0)
        ax.add_patch(rect)
        
        # Add group label at the top
        mid_x = start_x + width/2
        ax.text(mid_x, y_max * 0.97, group, 
                ha='center', va='bottom', 
                fontsize=12 * font_size_multiplier,
                fontweight='bold')
    

    # Customize the plot
    if metric == 'metric_score':
        ax.set_ylabel('AUROC Score', fontsize=12 * font_size_multiplier, fontweight='bold')
        title = f"Manipulation Detection (AUROC) {title_prefix}"
    else:
        ax.set_ylabel('AUROC Score', fontsize=12 * font_size_multiplier, fontweight='bold')
        title = f"{title_prefix}: Correctness (AUROC)"
    
    # ax.set_title(title, fontsize=15 * font_size_multiplier, fontweight='bold')
    ax.set_title(title, fontsize=15 * font_size_multiplier, fontweight='bold', pad=40)

    ax.set_xticks(x)
    
    # Format category labels using abbreviations
    labels = [category_display_names.get(cat, cat) for cat in categories]
    ax.set_xticklabels(labels, rotation=0, ha='center', 
                       fontsize=11 * font_size_multiplier)
    
    # Scale the font size of the y axis tick labels
    ax.tick_params(axis='y', labelsize=12 * font_size_multiplier)
    
        # Add legend with 4 entries for judge types (below the plot)
    judge_legend = ax.legend(
        fontsize=11 * font_size_multiplier,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.24),
        ncol=4,  # 4 columns for 4 legend entries
        borderaxespad=0,
    )
    
    

    # Add value labels on bars if requested
    if show_values:
        all_bars = [bars1, bars2, bars3, bars4]
        all_scores = [llm1_single_scores, llm1_steered_scores, llm2_single_scores, llm2_steered_scores]
        
        for bars, scores in zip(all_bars, all_scores):
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom', 
                           fontsize=8 * font_size_multiplier)
    
    # Add grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax. set(ylim=[0.5, 1.0])
    
    # Improve layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    return fig


# Code to run the combined comparison
def run_combined_comparison_Manip(leave_out_cats=None):
    """Run the combined LLM comparison using your existing data"""
    
    # Load data for Claude
    data_claude, _ = get_judge_result_file("manipulation", "Claude")
    responses_claude = data_claude['responses']
    
    # Load data for GPT4
    data_gpt4, _ = get_judge_result_file("manipulation", "GPT4Base")
    responses_gpt4 = data_gpt4['responses']
    
    # Calculate comparison data for both LLMs
    print("Calculating AUROC for Claude...")
    single_combined_claude, steered_combined_claude, comparison_pivot_claude = calculate_judge_comparison_auroc(
        responses_claude, metrics=['correctness', 'metric_score'], leave_out_cats=leave_out_cats
    )
    
    print("Calculating AUROC for GPT4...")
    single_combined_gpt4, steered_combined_gpt4, comparison_pivot_gpt4 = calculate_judge_comparison_auroc(
        responses_gpt4, metrics=['correctness', 'metric_score'], leave_out_cats=leave_out_cats
    )
    
    # Create combined plot
    fig = plot_combined_llm_comparison_Manip(
        comparison_pivot_claude, 
        comparison_pivot_gpt4,
        llm1_name="Claude-haiku",
        llm2_name="GPT4.1-Base", 
        metric='metric_score',
        figsize=(16, 8),
        font_size_multiplier=1.2,
        show_values=False,  # Set to True if you want values shown
        title_prefix="Combined LLM Comparison",
        leave_out_cats=leave_out_cats
    )
    
    plt.show()
    return fig
