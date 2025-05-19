import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict



def plot_summary_comparison(summary, font_size_multiplier=1.5, show_means=True):
    """
    Create a bar plot comparing base and suggestive responses across different evaluation types.
    
    Parameters:
    summary (dict): Summary statistics for each scenario
    font_size_multiplier (float): Factor to multiply all font sizes by (default: 1.0)
    show_means (bool): Whether to display the mean value above each bar (default: False)
    
    Returns:
    matplotlib.figure.Figure: The generated figure
    """
    # Extract data from summary - simplified data extraction
    data_configs = [
        ('Paired Evaluation', 'scenario2_paired'),
        ('Single Evaluation', 'scenario1_single'),
        ('JUSSA Evaluation', 'scenario3_steered_pairs')
    ]
    
    categories = []
    base_scores = []
    base_stds = []
    suggestive_scores = []
    suggestive_stds = []
    
    for label, key in data_configs:
        if key in summary:
            scenario = summary[key]
            base_scores.append(scenario['base']['mean'])
            base_stds.append(scenario['base']['std'])
            suggestive_scores.append(scenario['suggestive']['mean'])
            suggestive_stds.append(scenario['suggestive']['std'])
            categories.append(label)
    
    # Set up the figure with condensed code
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.35
    index = np.arange(len(categories))
    
    # Define styling parameters once
    styles = {
        'base': {'color': '#3274A1', 'label': 'Base Response'},
        'suggestive': {'color': '#E1812C', 'label': 'Provoked Response'}
    }
    
    # Create bars
    base_bars = ax.bar(index - bar_width/2, base_scores, bar_width, 
                       yerr=base_stds, capsize=5, alpha=0.8, **styles['base'])
    suggestive_bars = ax.bar(index + bar_width/2, suggestive_scores, bar_width, 
                             yerr=suggestive_stds, capsize=5, alpha=0.8, **styles['suggestive'])
    
    # Optionally add mean values above bars
    if show_means:
        for bars, scores in [(base_bars, base_scores), (suggestive_bars, suggestive_scores)]:
            for bar, score in zip(bars, scores):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, 
                        f'{score:.2f}', ha='center', va='bottom', fontsize=10 * font_size_multiplier)
    
    # Apply font sizes
    sizes = {
        'title': 14 * font_size_multiplier,
        'label': 16 * font_size_multiplier,
        'tick': 14 * font_size_multiplier,
        'legend': 14 * font_size_multiplier
    }
    
    # Configure plot
    ax.set_title('Base vs Provoked Response Scores Across Evaluation Types', 
                 fontsize=sizes['title'], pad=20)
    ax.set_ylabel('Score', fontsize=sizes['label'])
    ax.set_xticks(index)
    ax.set_xticklabels(categories, fontsize=sizes['tick'])
    ax.tick_params(axis='y', labelsize=sizes['tick'])
    ax.legend(fontsize=sizes['legend'])
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Set y-axis limits
    max_value = max(
        max(base_scores[i] + base_stds[i] for i in range(len(base_scores))),
        max(suggestive_scores[i] + suggestive_stds[i] for i in range(len(suggestive_scores)))
    )
    ax.set_ylim(0, max_value * 1.15)
    
    plt.tight_layout()
    return fig

def display_comparison_table(summary):
    """
    Display a simple table showing the comparison between base and suggestive responses.
    
    Parameters:
    summary (dict): Summary statistics for each scenario
    
    Returns:
    pandas.DataFrame: A formatted table with the comparison results
    """
    data = []
    
    # Define scenarios to check with their labels
    scenarios = [
        ('Paired Evaluation', 'scenario2_paired', 'gap'),
        ('Single Evaluation', 'scenario1_single', 'gap'),
        ('Non-steered Comparison', 'scenario3_steered_pairs', 'non_steered_comparison')
    ]
    
    for label, scenario_key, gap_key in scenarios:
        if scenario_key in summary:
            scenario = summary[scenario_key]
            
            # Get the appropriate gap data
            if gap_key in scenario:
                gap_data = scenario[gap_key]
                
                # Extract percentages based on available keys
                if 'suggestive_higher_percent' in gap_data:
                    suggestive_higher = f"{gap_data['suggestive_higher_percent']:.1f}%"
                    scores_equal = f"{gap_data.get('scores_equal_percent', 0.0):.1f}%"
                    base_higher = f"{gap_data.get('base_higher_percent', 0.0):.1f}%"
                elif 'positive_percent' in gap_data:
                    # For paired evaluation which might use different naming
                    suggestive_higher = f"{gap_data['positive_percent']:.1f}%"
                    scores_equal = "0.0%"
                    base_higher = f"{100 - gap_data['positive_percent']:.1f}%"
                else:
                    continue
                
                data.append([label, suggestive_higher, scores_equal, base_higher])
    
    # Create DataFrame
    df = pd.DataFrame(
        data,
        columns=[
            "Evaluation Type",
            "Suggestive > Base", 
            "Suggestive = Base", 
            "Base > Suggestive"
        ]
    )
    
    return df






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


def plot_difference_steered_single(statistics):
    """
    Plot the difference of mean steered evaluation and single evaluation for base and suggestive.

    Parameters:
    statistics (dict): Statistics dictionary from process_category_statistics

    Returns:
    matplotlib.figure.Figure: The generated figure
    """
    categories = sorted(statistics.keys())

    # Initialize data storage
    base_differences = []
    suggestive_differences = []

    for category in categories:
        if category in statistics:
            base_steered_mean = statistics[category].get('steered_base', {}).get('mean', 0)
            base_single_mean = statistics[category].get('single_base', {}).get('mean', 0)
            suggestive_steered_mean = statistics[category].get('steered_suggestive', {}).get('mean', 0)
            suggestive_single_mean = statistics[category].get('single_suggestive', {}).get('mean', 0)

            base_differences.append(base_steered_mean - base_single_mean)
            suggestive_differences.append(suggestive_steered_mean - suggestive_single_mean)

    # Plot the differences
    fig, ax = plt.subplots(figsize=(12, 6))
    x_positions = np.arange(len(categories))
    bar_width = 0.4

    ax.bar(x_positions - bar_width/2, base_differences, bar_width, label='Base', color='#3274A1', alpha=0.8)
    ax.bar(x_positions + bar_width/2, suggestive_differences, bar_width, label='Suggestive', color='#E1812C', alpha=0.8)

    ax.set_title('Difference of Mean Steered Evaluation and Single Evaluation', fontsize=14)
    ax.set_ylabel('Difference in Mean Scores', fontsize=12)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(categories, rotation=45, ha='center', fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    return fig

def analyze_steered_evaluation(responses):
    """
    Analyze correctness and preference patterns in steered evaluation experiments.
    
    Parameters:
    responses (list): List of response dictionaries containing judge results
    
    Returns:
    dict: Analysis results including correctness comparisons and preference rates
    """
    results = {
        'base_steered': {
            'non_steered_correctness': [],
            'steered_correctness': [],
            'steered_preferred_count': 0,
            'total_count': 0
        },
        'suggestive_steered': {
            'non_steered_correctness': [],
            'steered_correctness': [],
            'steered_preferred_count': 0,
            'total_count': 0
        }
    }
    
    for response in responses:
        # Base steered comparison
        if 'judge_base_steered_pair' in response:
            judge = response['judge_base_steered_pair']
            results['base_steered']['non_steered_correctness'].append(
                judge['response_A']['correctness'])
            results['base_steered']['steered_correctness'].append(
                judge['response_B']['correctness'])
            results['base_steered']['total_count'] += 1
            if judge['preferred_response'] == 'B':
                results['base_steered']['steered_preferred_count'] += 1
        
        # Suggestive steered comparison
        if 'judge_suggestive_steered_pair' in response:
            judge = response['judge_suggestive_steered_pair']
            results['suggestive_steered']['non_steered_correctness'].append(
                judge['response_A']['correctness'])
            results['suggestive_steered']['steered_correctness'].append(
                judge['response_B']['correctness'])
            results['suggestive_steered']['total_count'] += 1
            if judge['preferred_response'] == 'B':
                results['suggestive_steered']['steered_preferred_count'] += 1
    
    # Calculate statistics
    for key in results:
        if results[key]['total_count'] > 0:
            results[key]['steered_preference_rate'] = (
                results[key]['steered_preferred_count'] / results[key]['total_count'] * 100
            )
            results[key]['mean_non_steered_correctness'] = np.mean(
                results[key]['non_steered_correctness'])
            results[key]['mean_steered_correctness'] = np.mean(
                results[key]['steered_correctness'])
            results[key]['correctness_improvement'] = (
                results[key]['mean_steered_correctness'] - 
                results[key]['mean_non_steered_correctness']
            )
    
    return results


def analyze_evaluation_preferences(responses):
    """
    Analyze preference patterns and correctness across all evaluation types.
    
    Parameters:
    responses (list): List of response dictionaries containing judge results
    
    Returns:
    dict: Analysis results for single, paired, and steered evaluations
    """
    results = {
        'single': {
            'base_correctness': [],
            'suggestive_correctness': [],
            'suggestive_preferred_count': 0,
            'total_count': 0
        },
        'paired': {
            'base_correctness': [],
            'suggestive_correctness': [],
            'suggestive_preferred_count': 0,
            'total_count': 0
        },
        'steered': analyze_steered_evaluation(responses)
    }
    
    for response in responses:
        # Single evaluation analysis
        if 'judge_single' in response:
            single = response['judge_single']
            results['single']['base_correctness'].append(single['base']['correctness'])
            results['single']['suggestive_correctness'].append(single['suggestive']['correctness'])
            results['single']['total_count'] += 1
            # For single evaluation, we consider suggestive "preferred" if it has higher score
            if single['suggestive']['metric_score'] > single['base']['metric_score']:
                results['single']['suggestive_preferred_count'] += 1
        
        # Paired evaluation analysis
        if 'judge_base_vs_suggestive' in response:
            paired = response['judge_base_vs_suggestive']
            results['paired']['base_correctness'].append(paired['response_A']['correctness'])
            results['paired']['suggestive_correctness'].append(paired['response_B']['correctness'])
            results['paired']['total_count'] += 1
            if paired['preferred_response'] == 'B':
                results['paired']['suggestive_preferred_count'] += 1
    
    # Calculate preference rates
    for eval_type in ['single', 'paired']:
        if results[eval_type]['total_count'] > 0:
            results[eval_type]['suggestive_preference_rate'] = (
                results[eval_type]['suggestive_preferred_count'] / 
                results[eval_type]['total_count'] * 100
            )
            results[eval_type]['mean_base_correctness'] = np.mean(
                results[eval_type]['base_correctness'])
            results[eval_type]['mean_suggestive_correctness'] = np.mean(
                results[eval_type]['suggestive_correctness'])
    
    return results


def plot_correctness_comparison(results, figsize=(14, 6)):
    """
    Create visualization comparing correctness across evaluation types.
    
    Parameters:
    results (dict): Results from analyze_evaluation_preferences
    figsize (tuple): Figure size
    
    Returns:
    matplotlib.figure.Figure: The generated figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle('Correctness Comparison Across Evaluation Types', fontsize=16, y=1.02)
    
    # Single evaluation plot
    ax = axes[0]
    ax.set_title('Single Evaluation')
    single_data = results['single']
    ax.bar(['Base', 'Suggestive'], 
           [single_data['mean_base_correctness'], single_data['mean_suggestive_correctness']],
           color=['#3274A1', '#E1812C'], alpha=0.8)
    ax.set_ylabel('Mean Correctness')
    ax.set_ylim(0, 10)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Paired evaluation plot
    ax = axes[1]
    ax.set_title('Paired Evaluation')
    paired_data = results['paired']
    ax.bar(['Base', 'Suggestive'], 
           [paired_data['mean_base_correctness'], paired_data['mean_suggestive_correctness']],
           color=['#3274A1', '#E1812C'], alpha=0.8)
    ax.set_ylim(0, 10)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Steered evaluation plot
    ax = axes[2]
    ax.set_title('Steered Evaluation')
    steered_data = results['steered']
    
    categories = ['Base\n(Non-steered)', 'Base\n(Steered)', 'Suggestive\n(Non-steered)', 'Suggestive\n(Steered)']
    values = [
        steered_data['base_steered']['mean_non_steered_correctness'],
        steered_data['base_steered']['mean_steered_correctness'],
        steered_data['suggestive_steered']['mean_non_steered_correctness'],
        steered_data['suggestive_steered']['mean_steered_correctness']
    ]
    colors = ['#3274A1', '#5090C1', '#E1812C', '#F19E4C']
    
    ax.bar(categories, values, color=colors, alpha=0.8)
    ax.set_ylim(0, 10)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return fig


def print_preference_summary(results):
    """
    Print a comprehensive summary of preference patterns across all evaluation types.
    
    Parameters:
    results (dict): Results from analyze_evaluation_preferences
    """
    print("=== Preference Analysis Summary ===\n")
    
    print("Single Evaluation:")
    single = results['single']
    print(f"  Suggestive preferred: {single['suggestive_preference_rate']:.1f}%")
    print(f"  Base mean correctness: {single['mean_base_correctness']:.2f}")
    print(f"  Suggestive mean correctness: {single['mean_suggestive_correctness']:.2f}\n")
    
    print("Paired Evaluation:")
    paired = results['paired']
    print(f"  Suggestive preferred: {paired['suggestive_preference_rate']:.1f}%")
    print(f"  Base mean correctness: {paired['mean_base_correctness']:.2f}")
    print(f"  Suggestive mean correctness: {paired['mean_suggestive_correctness']:.2f}\n")
    
    print("Steered Evaluation:")
    steered = results['steered']
    
    print("  Base responses:")
    print(f"    Steered preferred over non-steered: {steered['base_steered']['steered_preference_rate']:.1f}%")
    print(f"    Non-steered correctness: {steered['base_steered']['mean_non_steered_correctness']:.2f}")
    print(f"    Steered correctness: {steered['base_steered']['mean_steered_correctness']:.2f}")
    print(f"    Improvement: {steered['base_steered']['correctness_improvement']:.2f}\n")
    
    print("  Suggestive responses:")
    print(f"    Steered preferred over non-steered: {steered['suggestive_steered']['steered_preference_rate']:.1f}%")
    print(f"    Non-steered correctness: {steered['suggestive_steered']['mean_non_steered_correctness']:.2f}")
    print(f"    Steered correctness: {steered['suggestive_steered']['mean_steered_correctness']:.2f}")
    print(f"    Improvement: {steered['suggestive_steered']['correctness_improvement']:.2f}")

# def plot_category_comparison(statistics, color_scheme='blue_orange', show_std=True, font_size_multiplier=1.0):
#     """
#     Plot all four evaluation types (base_single, base_steering, suggestive_single, suggestive_steering)
#     for each category with grouped bars.

#     Parameters:
#     statistics (dict): Statistics dictionary from process_category_statistics
#     color_scheme (str): Color scheme to use - 'blue_orange' or 'purple_green'
#     show_std (bool): Whether to include standard deviation as error bars
#     font_size_multiplier (float): Factor to scale all font sizes (default: 1.0)

#     Returns:
#     matplotlib.figure.Figure: The generated figure
#     """
#     categories = sorted(statistics.keys())
    
#     # Initialize data storage
#     base_single_values = []
#     base_steered_values = []
#     suggestive_single_values = []
#     suggestive_steered_values = []
#     base_single_stds = []
#     base_steered_stds = []
#     suggestive_single_stds = []
#     suggestive_steered_stds = []
    
#     for category in categories:
#         if category in statistics:
#             base_single_mean = statistics[category].get('single_base', {}).get('mean', 0)
#             base_steered_mean = statistics[category].get('steered_base', {}).get('mean', 0)
#             suggestive_single_mean = statistics[category].get('single_suggestive', {}).get('mean', 0)
#             suggestive_steered_mean = statistics[category].get('steered_suggestive', {}).get('mean', 0)
            
#             base_single_std = statistics[category].get('single_base', {}).get('std', 0)
#             base_steered_std = statistics[category].get('steered_base', {}).get('std', 0)
#             suggestive_single_std = statistics[category].get('single_suggestive', {}).get('std', 0)
#             suggestive_steered_std = statistics[category].get('steered_suggestive', {}).get('std', 0)
            
#             base_single_values.append(base_single_mean)
#             base_steered_values.append(base_steered_mean)
#             suggestive_single_values.append(suggestive_single_mean)
#             suggestive_steered_values.append(suggestive_steered_mean)
            
#             base_single_stds.append(base_single_std)
#             base_steered_stds.append(base_steered_std)
#             suggestive_single_stds.append(suggestive_single_std)
#             suggestive_steered_stds.append(suggestive_steered_std)
    
#     # Define color schemes - using darker colors for single, lighter for steered
#     if color_scheme == 'blue_orange':
#         colors = {
#             'base_single': '#2E75B6',      # Medium blue
#             'base_steered': '#AEC7E8',     # Light blue
#             'suggestive_single': '#D62728',   # Medium red-orange
#             'suggestive_steered': '#FF9896'   # Light red-orange
#         }
#     elif color_scheme == 'purple_green':
#         colors = {
#             'base_single': '#6B46C1',      # Medium purple
#             'base_steered': '#C084FC',     # Light purple
#             'suggestive_single': '#059669',   # Medium green
#             'suggestive_steered': '#6EE7B7'   # Light green
#         }
#     else:
#         # Default fallback
#         colors = {
#             'base_single': '#2E75B6',
#             'base_steered': '#AEC7E8',
#             'suggestive_single': '#D62728',
#             'suggestive_steered': '#FF9896'
#         }
    
#     # Create the plot
#     fig, ax = plt.subplots(figsize=(14, 8))
    
#     # Set up bar positions
#     bar_width = 0.18
#     category_gap = 0.5  # Increased gap between categories
#     group_gap = 0.03    # Small gap between base and suggestive groups within a category
    
#     # Calculate positions for all categories
#     x_positions = []
#     current_x = 0
    
#     for i in range(len(categories)):
#         x_positions.append(current_x)
#         if i < len(categories) - 1:
#             current_x += 4 * bar_width + group_gap + category_gap
    
#     x = np.array(x_positions)
    
#     # Calculate positions for each bar within a category
#     pos1 = x - 1.5 * bar_width - group_gap/2  # Base single
#     pos2 = x - 0.5 * bar_width - group_gap/2  # Base steered
#     pos3 = x + 0.5 * bar_width + group_gap/2  # Suggestive single
#     pos4 = x + 1.5 * bar_width + group_gap/2  # Suggestive steered
    
#     # Create bars with optional error bars
#     bars1 = ax.bar(pos1, base_single_values, bar_width, 
#                    label='Base Single', color=colors['base_single'], alpha=0.9,
#                    yerr=base_single_stds if show_std else None, capsize=5)
#     bars2 = ax.bar(pos2, base_steered_values, bar_width,
#                    label='Base JUSSA', color=colors['base_steered'], alpha=0.9,
#                    yerr=base_steered_stds if show_std else None, capsize=5)
#     bars3 = ax.bar(pos3, suggestive_single_values, bar_width,
#                    label='Provoked Single', color=colors['suggestive_single'], alpha=0.9,
#                    yerr=suggestive_single_stds if show_std else None, capsize=5)
#     bars4 = ax.bar(pos4, suggestive_steered_values, bar_width,
#                    label='Provoked JUSSA', color=colors['suggestive_steered'], alpha=0.9,
#                    yerr=suggestive_steered_stds if show_std else None, capsize=5)
    
#     # Customize the plot
#     ax.set_xlabel('Categories', fontsize=12 * font_size_multiplier)
#     ax.set_ylabel('Mean Scores', fontsize=12 * font_size_multiplier)
#     ax.set_title('Comparison of All Evaluation Types by Category', fontsize=15 * font_size_multiplier, fontweight='bold')
#     ax.set_xticks(x)
#     ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=12 * font_size_multiplier)
    
#     # Scale the font size of the y axis tick labels (numbers)
#     ax.tick_params(axis='y', labelsize=12 * font_size_multiplier)
    
#     # Add legend with better positioning
#     ax.legend(fontsize=12 * font_size_multiplier, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=1)
    
#     # Add grid for better readability
#     ax.grid(axis='y', linestyle='--', alpha=0.3)
#     ax.set_axisbelow(True)
    
#     # Add value labels on bars (optional)
#     def add_value_labels(bars, fontsize=8):
#         for bar in bars:
#             height = bar.get_height()
#             if height > 0:
#                 ax.annotate(f'{height:.2f}',
#                            xy=(bar.get_x() + bar.get_width() / 2, height),
#                            xytext=(0, 3),  # 3 points vertical offset
#                            textcoords="offset points",
#                            ha='center', va='bottom',
#                            fontsize=fontsize * font_size_multiplier)
    
#     # Add labels if values are not too dense
#     if len(categories) <= 8:
#         add_value_labels(bars1)
#         add_value_labels(bars2)
#         add_value_labels(bars3)
#         add_value_labels(bars4)
    
#     plt.tight_layout()
#     return fig

def plot_category_comparison(statistics, category_groups=None, color_scheme='blue_orange', show_std=True, font_size_multiplier=1.0, fig_size=(16, 8)):
    """
    Plot all four evaluation types for each category with grouped bars,
    with categories organized by their higher-level category groups.

    Parameters:
    statistics (dict): Statistics dictionary from process_category_statistics
    category_groups (dict): Mapping of categories to their higher-level groups
                           If None, default grouping will be used
    color_scheme (str): Color scheme to use - 'blue_orange' or 'purple_green'
    show_std (bool): Whether to include standard deviation as error bars
    font_size_multiplier (float): Factor to scale all font sizes

    Returns:
    matplotlib.figure.Figure: The generated figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Rectangle
    
    # Define default category grouping if not provided
    if category_groups is None:
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
    
    # Group categories and sort within groups
    available_categories = [cat for cat in statistics.keys() if cat in category_groups]
    
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
    
    # Initialize data storage
    base_single_values = []
    base_steered_values = []
    suggestive_single_values = []
    suggestive_steered_values = []
    base_single_stds = []
    base_steered_stds = []
    suggestive_single_stds = []
    suggestive_steered_stds = []
    
    for category in categories:
        if category in statistics:
            base_single_mean = statistics[category].get('single_base', {}).get('mean', 0)
            base_steered_mean = statistics[category].get('steered_base', {}).get('mean', 0)
            suggestive_single_mean = statistics[category].get('single_suggestive', {}).get('mean', 0)
            suggestive_steered_mean = statistics[category].get('steered_suggestive', {}).get('mean', 0)
            
            base_single_std = statistics[category].get('single_base', {}).get('std', 0)
            base_steered_std = statistics[category].get('steered_base', {}).get('std', 0)
            suggestive_single_std = statistics[category].get('single_suggestive', {}).get('std', 0)
            suggestive_steered_std = statistics[category].get('steered_suggestive', {}).get('std', 0)
            
            base_single_values.append(base_single_mean)
            base_steered_values.append(base_steered_mean)
            suggestive_single_values.append(suggestive_single_mean)
            suggestive_steered_values.append(suggestive_steered_mean)
            
            base_single_stds.append(base_single_std)
            base_steered_stds.append(base_steered_std)
            suggestive_single_stds.append(suggestive_single_std)
            suggestive_steered_stds.append(suggestive_steered_std)
    
    # Define color schemes - using darker colors for single, lighter for steered
    if color_scheme == 'blue_orange':
        colors = {
            'base_single': '#2E75B6',      # Medium blue
            'base_steered': '#AEC7E8',     # Light blue
            'suggestive_single': '#D62728',   # Medium red-orange
            'suggestive_steered': '#FF9896'   # Light red-orange
        }
        group_colors = {
            'Skewed Presentation': '#E6F2FF',  # Very light blue
            'Misleading Claims': '#FFF2E6',    # Very light orange
            'Emotional Pressure': '#F2FFE6'    # Very light green
        }
    elif color_scheme == 'purple_green':
        colors = {
            'base_single': '#6B46C1',      # Medium purple
            'base_steered': '#C084FC',     # Light purple
            'suggestive_single': '#059669',   # Medium green
            'suggestive_steered': '#6EE7B7'   # Light green
        }
        group_colors = {
            'Skewed Presentation': '#F3EBFF',  # Very light purple
            'Misleading Claims': '#EBFFF3',    # Very light green
            'Emotional Pressure': '#FFF3EB'    # Very light orange
        }
    else:
        # Default fallback
        colors = {
            'base_single': '#2E75B6',
            'base_steered': '#AEC7E8',
            'suggestive_single': '#D62728',
            'suggestive_steered': '#FF9896'
        }
        group_colors = {
            'Skewed Presentation': '#E6F2FF',
            'Misleading Claims': '#FFF2E6',
            'Emotional Pressure': '#F2FFE6'
        }
    
    # Create the plot
    fig, ax = plt.subplots(figsize=fig_size)  # Increased size for better readability
    
    # Set up bar positions
    # bar_width = 0.18
    bar_width = 0.22
    category_gap = 0.3  
    group_gap = 0.03    
    extra_group_gap = 0.7  # Additional gap between different higher-level groups
    
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
        current_x += 4 * bar_width + group_gap + category_gap
        last_group = group
    
    x = np.array(x_positions)
    
    # Calculate positions for each bar within a category
    pos1 = x - 1.5 * bar_width - group_gap/2  # Base single
    pos2 = x - 0.5 * bar_width - group_gap/2  # Base steered
    pos3 = x + 0.5 * bar_width + group_gap/2  # Suggestive single
    pos4 = x + 1.5 * bar_width + group_gap/2  # Suggestive steered
    
    # Create bars with optional error bars
    bars1 = ax.bar(pos1, base_single_values, bar_width, 
                   label='Base Single', color=colors['base_single'], alpha=0.9,
                   yerr=base_single_stds if show_std else None, capsize=5)
    bars2 = ax.bar(pos2, base_steered_values, bar_width,
                   label='Base JUSSA', color=colors['base_steered'], alpha=0.9,
                   yerr=base_steered_stds if show_std else None, capsize=5)
    bars3 = ax.bar(pos3, suggestive_single_values, bar_width,
                   label='Provoked Single', color=colors['suggestive_single'], alpha=0.9,
                   yerr=suggestive_single_stds if show_std else None, capsize=5)
    bars4 = ax.bar(pos4, suggestive_steered_values, bar_width,
                   label='Provoked JUSSA', color=colors['suggestive_steered'], alpha=0.9,
                   yerr=suggestive_steered_stds if show_std else None, capsize=5)
    
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
                end_x = x[i-1] + 2 * bar_width + group_gap/2
                group_spans[last_group] = (group_start_x, end_x)
            
            # Start of a new group
            group_start_x = x[i] - 2 * bar_width - group_gap/2
            last_group = group
    
    # Add the last group
    if last_group is not None:
        end_x = x[-1] + 2 * bar_width + group_gap/2
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
    # ax.set_xlabel('Categories', fontsize=12 * font_size_multiplier)
    ax.set_ylabel('Mean Scores', fontsize=12 * font_size_multiplier)
    ax.set_title('Comparison of Evaluation Types by Category and Group', 
                fontsize=15 * font_size_multiplier, fontweight='bold')
    ax.set_xticks(x)
    #  prev version:

    labels = [cat.replace('_', ' ').title() for cat in categories]

    ax.set_xticklabels(labels, 
                    rotation=-25, ha='left', fontsize=12 * font_size_multiplier)    


    # Scale the font size of the y axis tick labels (numbers)
    ax.tick_params(axis='y', labelsize=12 * font_size_multiplier)
    

    ax.legend(
        fontsize=12 * font_size_multiplier,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.19),
        ncol=4,
        borderaxespad=0
    )

    # Add grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    # Add value labels on bars if not too crowded
    def add_value_labels(bars, fontsize=8):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=fontsize * font_size_multiplier)
    
    # Add labels if values are not too dense
    if len(categories) <= 10:
        add_value_labels(bars1)
        add_value_labels(bars2)
        add_value_labels(bars3)
        add_value_labels(bars4)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # set y limit to 8
    # ax.set_ylim(0, 8)
    # ax.set_ylim(0, 10)
    
    plt.tight_layout()
    # plt.tight_layout(rect=[0, 0, 0.85, 1])
    return fig


# if __name__ == "__main__":

#     fig = plot_summary_comparison(summary, font_size_multiplier=1.5)

#     # Display comparison table
#     table = display_comparison_table(summary)
#     print("\nComparison Table:")
#     # display(table)


#     ##### Plot results per category #####
#     # Process statistics
#     cat_param="higher_level_category"
#     # cat_param="category_id"

#     metric_name='metric_score'
#     # metric_name='correctness'


#     statistics = process_category_statistics(data['responses'], cat_param=cat_param, metric_name=metric_name)

#     figsize=(14,8)
#     # figsize=(20,6)
#     # Create visualization
#     fig = visualize_category_statistics(statistics, figsize=figsize)
#     plt.show()


#     # Create summary table
#     summary_table = create_category_summary_table(statistics)
#     print("\nCategory Summary:")
#     print(summary_table)



#     fig = plot_difference_steered_single(statistics)
#     plt.show()

#     # Example usage:
#     # Using the default blue/orange color scheme
#     fig = plot_category_comparison(statistics)
#     plt.show()



#     # Perform comprehensive analysis
#     results = analyze_evaluation_preferences(data['responses'])

#     # Print summary
#     print_preference_summary(results)

#     # Create visualization
#     fig = plot_correctness_comparison(results)
#     plt.show()