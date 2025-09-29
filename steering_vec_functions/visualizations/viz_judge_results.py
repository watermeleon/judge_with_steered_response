import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import json

# Category groupings and display names (same as original)
category_groups = {
    'cherry_picked_evidence': 'Skewed Presentation',
    'false_dichotomy': 'Skewed Presentation',
    'false_transparency': 'Skewed Presentation',
    'assistant_self_interest': 'Skewed Presentation',
    'strategic_omission': 'Skewed Presentation',
    'risk_distortion': 'Misleading Claims',
    'false_causality': 'Misleading Claims',
    'manufactured_consensus': 'Misleading Claims',
    'false_credentials': 'Misleading Claims',
    'hidden_premises': 'Misleading Claims',
    'emotional_manipulation': 'Emotional Pressure',
    'manufactured_scarcity': 'Emotional Pressure',
    'artificial_time_pressure': 'Emotional Pressure'
}

category_display_names = {
    'cherry_picked_evidence': 'Cherry Picked Evidence',
    'false_dichotomy': 'False Dichotomy', 
    'false_transparency': 'False Transparency',
    'assistant_self_interest': 'Assistant Self Interest',
    'strategic_omission': 'Strategic Omission',
    'risk_distortion': 'Risk Distortion',
    'false_causality': 'False Causality',
    'manufactured_consensus': 'Manufactured Consensus',
    'false_credentials': 'False Credentials',
    'hidden_premises': 'Hidden Premises',
    'emotional_manipulation': 'Emotional Manipulation',
    'manufactured_scarcity': 'Manufactured Scarcity',
    'artificial_time_pressure': 'Artificial Time Pressure'
}


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
    fig, ax = plt.subplots(figsize=(8, 4))
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


def plot_category_comparison(statistics, category_groups=None, color_scheme='blue_orange', show_std=True, font_size_multiplier=1.0, fig_size=(16, 8), leave_out_cats=None):
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

    if leave_out_cats is not None:
        category_groups = {cat: group for cat, group in category_groups.items() if cat not in leave_out_cats}

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
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    return fig


def create_judge_preference_stacked_bars_single(enhanced_df_gpt, enhanced_df_claude, 
                                              metric='preference', font_scale=1.0, ax=None, 
                                              bar_height=0.85, show_y_labels=True, show_legend=True,
                                              judge_labels=None, figsize=(4, 8)):
    """
    Create a single plot showing horizontal stacked bars for judge results across manipulation categories.
    This is a modular version of the main function for use in multi-panel figures.
    
    Parameters:
    enhanced_df_gpt (pd.DataFrame): Enhanced results DataFrame for GPT judge
    enhanced_df_claude (pd.DataFrame): Enhanced results DataFrame for Claude judge
    metric (str): Which metric to visualize ('preference', 'correctness', 'manipulation')
    font_scale (float): Scaling factor for all font sizes (default: 1.0)
    ax (matplotlib.axes.Axes): Axis to plot on (if None, creates new figure)
    bar_height (float): Height of bars as fraction of available space (0.0-1.0)
    show_y_labels (bool): Whether to show category labels on y-axis
    show_legend (bool): Whether to show legend
    judge_labels (list): Custom labels for judges (default: ['GPT4.1-nano', 'Claude3.5-haiku'])
    
    Returns:
    tuple: (fig, ax) matplotlib figure and axis objects
    """
    
    # Define metric mappings (same as original)
    metric_configs = {
        'preference': {
            'columns': ['base_preferred_pct', 'equal_preferred_pct', 'provoked_preferred_pct'],
            'labels': ['Base Preferred', 'Equal/Tie', 'Provoked Preferred'],
            'title': 'Preference Distribution',
            'colors': {
                'first': '#7B1FA2',    # Purple for base higher
                'second': "#F5852F",   # Orange for equal
                'third': "#1B83AF"     # Dark orange for base lower
            }
        },
        'correctness': {
            'columns': ['correctness_base_higher_pct', 'correctness_equal_pct', 'correctness_base_lower_pct'],
            'labels': ['Base Higher', 'Equal', 'Base Lower'],
            'title': 'Correctness Distribution',
            'colors': {
                'first': '#7B1FA2',    # Purple for base higher
                'second': "#F5852F",   # Orange for equal
                'third': "#1B83AF"     # Dark orange for base lower
            }
        },
        'manipulation': {
            'columns': ['metric_base_higher_pct', 'metric_equal_pct', 'metric_base_lower_pct'],
            'labels': ['Base Higher', 'Equal', 'Base Lower'],
            'title': 'Manipulation Distribution',
            'colors': {
                'first': '#7B1FA2',    # Purple for base higher
                'second': "#F5852F",   # Orange for equal
                'third': "#1B83AF"     # Dark orange for base lower
            }
        }
    }
    
    if metric not in metric_configs:
        raise ValueError(f"metric must be one of {list(metric_configs.keys())}")
    
    config = metric_configs[metric]
    

    
    # Create lookup dictionaries
    gpt_lookup = {row['category']: row for _, row in enhanced_df_gpt.iterrows()}
    claude_lookup = {row['category']: row for _, row in enhanced_df_claude.iterrows()}
    
    # Group and sort categories
    grouped_categories = {}
    for category, group in category_groups.items():
        if group not in grouped_categories:
            grouped_categories[group] = []
        grouped_categories[group].append(category)
    
    # Order groups and categories
    group_order = ['Skewed Presentation', 'Misleading Claims', 'Emotional Pressure']
    ordered_categories = []
    for group in group_order:
        if group in grouped_categories:
            sorted_cats = sorted(grouped_categories[group])
            ordered_categories.extend(sorted_cats)
    
    # Filter categories that exist in both datasets
    valid_categories = []
    category_labels = []
    
    for category in ordered_categories:
        if category in gpt_lookup and category in claude_lookup:
            valid_categories.append(category)
            category_labels.append(category_display_names.get(category, category))
    
    # Create figure if ax not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # Set up the grid
    n_categories = len(valid_categories)
    n_judges = 2
    
    # Create the grid background
    ax.set_xlim(0, n_judges)
    ax.set_ylim(0, n_categories)
    
    # Draw horizontal stacked bars for each cell
    bar_margin = (1 - bar_height) / 2
    
    for i, category in enumerate(valid_categories):
        y_pos = n_categories - i - 1  # Flip to match top-to-bottom ordering
        
        # GPT judge (column 0)
        if category in gpt_lookup:
            gpt_row = gpt_lookup[category]
            # Get percentages for the selected metric
            first_pct = gpt_row[config['columns'][0]]
            second_pct = gpt_row[config['columns'][1]]
            third_pct = gpt_row[config['columns'][2]]
            
            # Normalize to ensure they sum to 100 (in case of rounding errors)
            total = first_pct + second_pct + third_pct
            if total > 0:
                first_pct = (first_pct / total) * 100
                second_pct = (second_pct / total) * 100
                third_pct = (third_pct / total) * 100
            
            # Draw horizontal stacked bar for GPT
            y_center = y_pos + 0.5
            y_bottom = y_center - bar_height/2
            
            # First segment (left)
            if first_pct > 0:
                rect = patches.Rectangle((0, y_bottom), first_pct/100, bar_height, 
                                        facecolor=config['colors']['first'], edgecolor='white', linewidth=0.5)
                ax.add_patch(rect)
            
            # Second segment (middle)
            if second_pct > 0:
                rect = patches.Rectangle((first_pct/100, y_bottom), second_pct/100, bar_height,
                                        facecolor=config['colors']['second'], edgecolor='white', linewidth=0.5)
                ax.add_patch(rect)
            
            # Third segment (right)
            if third_pct > 0:
                rect = patches.Rectangle(((first_pct + second_pct)/100, y_bottom), third_pct/100, bar_height,
                                        facecolor=config['colors']['third'], edgecolor='white', linewidth=0.5)
                ax.add_patch(rect)
        
        # Claude judge (column 1)
        if category in claude_lookup:
            claude_row = claude_lookup[category]
            # Get percentages for the selected metric
            first_pct = claude_row[config['columns'][0]]
            second_pct = claude_row[config['columns'][1]]
            third_pct = claude_row[config['columns'][2]]
            
            # Normalize to ensure they sum to 100 (in case of rounding errors)
            total = first_pct + second_pct + third_pct
            if total > 0:
                first_pct = (first_pct / total) * 100
                second_pct = (second_pct / total) * 100
                third_pct = (third_pct / total) * 100
            
            # Draw horizontal stacked bar for Claude
            y_center = y_pos + 0.5
            y_bottom = y_center - bar_height/2
            
            # First segment (left)
            if first_pct > 0:
                rect = patches.Rectangle((1, y_bottom), first_pct/100, bar_height,
                                        facecolor=config['colors']['first'], edgecolor='white', linewidth=0.5)
                ax.add_patch(rect)
            
            # Second segment (middle)
            if second_pct > 0:
                rect = patches.Rectangle((1 + first_pct/100, y_bottom), second_pct/100, bar_height,
                                        facecolor=config['colors']['second'], edgecolor='white', linewidth=0.5)
                ax.add_patch(rect)
            
            # Third segment (right)
            if third_pct > 0:
                rect = patches.Rectangle((1 + (first_pct + second_pct)/100, y_bottom), third_pct/100, bar_height,
                                        facecolor=config['colors']['third'], edgecolor='white', linewidth=0.5)
                ax.add_patch(rect)
    
    # Add grid lines
    for i in range(n_categories + 1):
        ax.axhline(y=i, color='gray', linewidth=0.5, alpha=0.3)
    ax.axvline(x=1, color='white', linewidth=3)  # Separator between judges
    
    # Add group separators
    group_boundaries = []
    current_pos = 0
    
    for group in group_order:
        if group in grouped_categories:
            group_size = len([cat for cat in grouped_categories[group] 
                            if cat in gpt_lookup and cat in claude_lookup])
            current_pos += group_size
            if current_pos < n_categories:
                group_boundaries.append(n_categories - current_pos)
    
    for boundary in group_boundaries:
        ax.axhline(y=boundary, color='white', linewidth=3)
    
    # Set title
    ax.set_title(config['title'], fontsize=12 * font_scale, fontweight='bold', pad=15)
    
    # X-axis labels (judge names)
    if judge_labels is None:
        judge_labels = ['GPT4.1-nano', 'Claude3.5-haiku']
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(judge_labels, fontsize=10 * font_scale, fontweight='bold')
    
    # Y-axis labels (categories) - only show if requested
    if show_y_labels:
        print("show_y_labels", show_y_labels)
        ax.set_yticks([i + 0.5 for i in range(n_categories)])
        ax.set_yticklabels(category_labels[::-1], fontsize=9 * font_scale)
    # else:
    #     ax.set_yticks([])
    #     ax.set_yticklabels([])
    
    # Remove axis spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Remove tick marks
    ax.tick_params(axis='both', which='both', length=0)
    
    # Add legend only if requested
    if show_legend:
        legend_elements = [
            patches.Patch(facecolor=config['colors']['first'], label=config['labels'][0]),
            patches.Patch(facecolor=config['colors']['second'], label=config['labels'][1]),
            patches.Patch(facecolor=config['colors']['third'], label=config['labels'][2])
        ]
        ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.08),
                 ncol=3, fontsize=8 * font_scale, frameon=False)
    
    return fig, ax


def create_combined_judge_preference_plots(enhanced_df_gpt, enhanced_df_claude, 
                                         font_scale=1.0, figsize=(18, 8), 
                                         bar_height=0.85, show_group_labels=True):
    """
    Create a combined visualization showing all three metrics side by side.
    Category names are shown only on the leftmost plot for a clean appearance.
    
    Parameters:
    enhanced_df_gpt (pd.DataFrame): Enhanced results DataFrame for GPT judge
    enhanced_df_claude (pd.DataFrame): Enhanced results DataFrame for Claude judge
    font_scale (float): Scaling factor for all font sizes (default: 1.0)
    figsize (tuple): Figure size (width, height) in inches
    bar_height (float): Height of bars as fraction of available space (0.0-1.0)
    show_group_labels (bool): Whether to show group labels on the right side
    
    Returns:
    tuple: (fig, axes) matplotlib figure and list of axis objects
    """
    
    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
    
    # Define metrics to plot
    metrics = ['preference', 'correctness', 'manipulation']
    
    # Create each subplot
    for i, metric in enumerate(metrics):
        show_y_labels = (i == 0)  # Only show y-labels on the first (leftmost) plot
        show_legend = (i == 1)    # Only show legend on the middle plot
        print("show_y_labels", show_y_labels)
        
        create_judge_preference_stacked_bars_single(
            enhanced_df_gpt, enhanced_df_claude,
            metric=metric,
            font_scale=font_scale,
            ax=axes[i],
            bar_height=bar_height,
            show_y_labels=show_y_labels,
            show_legend=show_legend
        )
    
    # Add group labels on the right side of the rightmost plot if requested
    if show_group_labels:
        
        # Create lookup dictionaries
        gpt_lookup = {row['category']: row for _, row in enhanced_df_gpt.iterrows()}
        claude_lookup = {row['category']: row for _, row in enhanced_df_claude.iterrows()}
        
        # Group and sort categories
        grouped_categories = {}
        for category, group in category_groups.items():
            if group not in grouped_categories:
                grouped_categories[group] = []
            grouped_categories[group].append(category)
        
        # Order groups and categories
        group_order = ['Skewed Presentation', 'Misleading Claims', 'Emotional Pressure']
        ordered_categories = []
        for group in group_order:
            if group in grouped_categories:
                sorted_cats = sorted(grouped_categories[group])
                ordered_categories.extend(sorted_cats)
        
        # Filter categories that exist in both datasets
        valid_categories = []
        for category in ordered_categories:
            if category in gpt_lookup and category in claude_lookup:
                valid_categories.append(category)
        
        n_categories = len(valid_categories)
        
        # Add group labels
        current_pos = 0
        for group in group_order:
            if group in grouped_categories:
                group_size = len([cat for cat in grouped_categories[group] 
                                if cat in gpt_lookup and cat in claude_lookup])
                group_center = n_categories - current_pos - group_size / 2
                axes[2].text(2.15, group_center, group.replace(" ", "\n"),
                           rotation=90, va='center', ha='left',
                           fontsize=10 * font_scale, fontweight='bold')
                current_pos += group_size
    
    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.05)

    
    plt.tight_layout()
    
    return fig, axes

def extract_per_question_judgments(data):
    """
    Extract judgments for each question and each metric from a single judge's data.
    
    Parameters:
    data (dict): Dictionary containing 'responses' list from a single judge
    
    Returns:
    dict: Nested dictionary with structure {question_id: {metric: judgment}}
          where judgment is 'base', 'provoked', or 'equal'
    """
    
    question_judgments = {}
    responses = data['responses']
    
    for idx, response in enumerate(responses):
        # Skip if no pair judge data
        if 'judge_base_vs_suggestive' not in response:
            continue
        
        # Use index as question_id if no other identifier is available
        # You might want to use a different identifier if available in your data
        question_id = idx  # or response.get('question_id', idx) if available
        
        pair_judge = response['judge_base_vs_suggestive']
        
        # Initialize judgments for this question
        question_judgments[question_id] = {
            'category': response['full_data']['category_id']
        }
        
        # 1. Preference judgment
        if pair_judge['preferred_response'] == 'A':
            question_judgments[question_id]['preference'] = 'base'
        elif pair_judge['preferred_response'] == 'B':
            question_judgments[question_id]['preference'] = 'provoked'
        else:  # None or tie
            question_judgments[question_id]['preference'] = 'equal'
        
        # 2. Correctness judgment
        base_correctness = pair_judge['response_A']['correctness']
        suggestive_correctness = pair_judge['response_B']['correctness']
        
        if base_correctness > suggestive_correctness:
            question_judgments[question_id]['correctness'] = 'base'
        elif base_correctness < suggestive_correctness:
            question_judgments[question_id]['correctness'] = 'provoked'
        else:
            question_judgments[question_id]['correctness'] = 'equal'
        
        # 3. Manipulation/Metric judgment
        base_metric = pair_judge['response_A']['metric_score']
        suggestive_metric = pair_judge['response_B']['metric_score']
        
        if base_metric > suggestive_metric:
            question_judgments[question_id]['manipulation'] = 'base'
        elif base_metric < suggestive_metric:
            question_judgments[question_id]['manipulation'] = 'provoked'
        else:
            question_judgments[question_id]['manipulation'] = 'equal'
    
    return question_judgments


def combine_two_judge_decisions(judge1_decision, judge2_decision):
    """
    Combine decisions from two judges for a single metric.
    
    Rules:
    - If both agree (both 'base' or both 'provoked') → that answer
    - If they disagree (one 'base', one 'provoked') → 'equal' (tie)
    - If one is 'equal' and other has preference → the preference wins
    - If both are 'equal' → 'equal'
    
    Parameters:
    judge1_decision (str): 'base', 'provoked', or 'equal'
    judge2_decision (str): 'base', 'provoked', or 'equal'
    
    Returns:
    str: Combined decision ('base', 'provoked', or 'equal')
    """
    
    # Both agree on the same preference
    if judge1_decision == judge2_decision:
        return judge1_decision
    
    # One says equal, the other has a preference
    if judge1_decision == 'equal':
        return judge2_decision
    if judge2_decision == 'equal':
        return judge1_decision
    
    # They disagree (one says base, other says provoked)
    return 'equal'


def combine_judge_results(data_judge1, data_judge2):
    """
    Combine results from two judges on a per-question basis.
    
    Parameters:
    data_judge1 (dict): Data from first judge
    data_judge2 (dict): Data from second judge
    
    Returns:
    dict: Combined judgments per question
    """
    
    # Extract per-question judgments for each judge
    judge1_judgments = extract_per_question_judgments(data_judge1)
    judge2_judgments = extract_per_question_judgments(data_judge2)
    
    # Find common questions (both judges must have evaluated the same questions)
    common_questions = set(judge1_judgments.keys()) & set(judge2_judgments.keys())
    
    combined_judgments = {}
    
    for q_id in common_questions:
        j1 = judge1_judgments[q_id]
        j2 = judge2_judgments[q_id]
        
        # Ensure both judges evaluated the same category
        if j1['category'] != j2['category']:
            print(f"Warning: Category mismatch for question {q_id}")
            continue
        
        combined_judgments[q_id] = {
            'category': j1['category'],
            'preference': combine_two_judge_decisions(j1['preference'], j2['preference']),
            'correctness': combine_two_judge_decisions(j1['correctness'], j2['correctness']),
            'manipulation': combine_two_judge_decisions(j1['manipulation'], j2['manipulation'])
        }
    
    return combined_judgments


def aggregate_combined_results_by_category(combined_judgments):
    """
    Aggregate combined judge results by category.
    
    Parameters:
    combined_judgments (dict): Combined judgments from combine_judge_results
    
    Returns:
    pd.DataFrame: Aggregated results per category with counts and percentages
    """
    
    # Initialize category statistics
    category_stats = defaultdict(lambda: {
        'total_count': 0,
        # Preference
        'preference_base': 0,
        'preference_equal': 0,
        'preference_provoked': 0,
        # Correctness
        'correctness_base': 0,
        'correctness_equal': 0,
        'correctness_provoked': 0,
        # Manipulation
        'manipulation_base': 0,
        'manipulation_equal': 0,
        'manipulation_provoked': 0
    })
    
    # Count occurrences
    for q_id, judgment in combined_judgments.items():
        cat = judgment['category']
        category_stats[cat]['total_count'] += 1
        
        # Count preference
        if judgment['preference'] == 'base':
            category_stats[cat]['preference_base'] += 1
        elif judgment['preference'] == 'equal':
            category_stats[cat]['preference_equal'] += 1
        else:  # provoked
            category_stats[cat]['preference_provoked'] += 1
        
        # Count correctness
        if judgment['correctness'] == 'base':
            category_stats[cat]['correctness_base'] += 1
        elif judgment['correctness'] == 'equal':
            category_stats[cat]['correctness_equal'] += 1
        else:  # provoked
            category_stats[cat]['correctness_provoked'] += 1
        
        # Count manipulation
        if judgment['manipulation'] == 'base':
            category_stats[cat]['manipulation_base'] += 1
        elif judgment['manipulation'] == 'equal':
            category_stats[cat]['manipulation_equal'] += 1
        else:  # provoked
            category_stats[cat]['manipulation_provoked'] += 1
    
    # Convert to DataFrame
    df_data = []
    for category, stats in category_stats.items():
        row = {
            'category': category,
            'total_responses': stats['total_count'],
            # Raw counts
            'preference_base': stats['preference_base'],
            'preference_equal': stats['preference_equal'],
            'preference_provoked': stats['preference_provoked'],
            'correctness_base': stats['correctness_base'],
            'correctness_equal': stats['correctness_equal'],
            'correctness_provoked': stats['correctness_provoked'],
            'manipulation_base': stats['manipulation_base'],
            'manipulation_equal': stats['manipulation_equal'],
            'manipulation_provoked': stats['manipulation_provoked']
        }
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    df = df.sort_values('category').reset_index(drop=True)
    
    # Add percentage columns
    for metric in ['preference', 'correctness', 'manipulation']:
        for outcome in ['base', 'equal', 'provoked']:
            col_name = f'{metric}_{outcome}'
            pct_col_name = f'{col_name}_pct'
            df[pct_col_name] = (df[col_name] / df['total_responses'] * 100).round(1)
    
    return df


def prepare_combined_results_for_visualization(combined_df):
    """
    Prepare the combined results DataFrame for use with the existing visualization functions.
    This renames columns to match the expected format.
    
    Parameters:
    combined_df (pd.DataFrame): Output from aggregate_combined_results_by_category
    
    Returns:
    pd.DataFrame: DataFrame with columns renamed for visualization compatibility
    """
    
    # Create a copy to avoid modifying the original
    viz_df = combined_df.copy()
    
    # Rename columns for visualization compatibility
    viz_df['base_preferred_pct'] = viz_df['preference_base_pct']
    viz_df['equal_preferred_pct'] = viz_df['preference_equal_pct']
    viz_df['provoked_preferred_pct'] = viz_df['preference_provoked_pct']
    
    viz_df['correctness_base_higher_pct'] = viz_df['correctness_base_pct']
    viz_df['correctness_equal_pct'] = viz_df['correctness_equal_pct']
    viz_df['correctness_base_lower_pct'] = viz_df['correctness_provoked_pct']
    
    viz_df['metric_base_higher_pct'] = viz_df['manipulation_base_pct']
    viz_df['metric_equal_pct'] = viz_df['manipulation_equal_pct']
    viz_df['metric_base_lower_pct'] = viz_df['manipulation_provoked_pct']
    
    return viz_df


# Example usage:
def process_and_visualize_combined_judges(data_gpt, data_claude, verbose=True):
    """
    Complete pipeline to combine judge results and prepare for visualization.
    
    Parameters:
    data_gpt (dict): GPT judge data
    data_claude (dict): Claude judge data
    verbose (bool): If True, print summary statistics
    
    Returns:
    pd.DataFrame: Combined and aggregated results ready for visualization
    """
    
    # Step 1: Combine judge results on per-question basis
    combined_judgments = combine_judge_results(data_gpt, data_claude)
    
    # Step 2: Aggregate by category
    combined_df = aggregate_combined_results_by_category(combined_judgments)
    
    # Step 3: Prepare for visualization
    viz_ready_df = prepare_combined_results_for_visualization(combined_df)
    
    if verbose:
        print("\n=== COMBINED JUDGE RESULTS SUMMARY ===")
        print(f"Total questions evaluated by both judges: {len(combined_judgments)}")
        print(f"Number of categories: {len(combined_df)}")
        print("\nCategory breakdown:")
        print(combined_df[['category', 'total_responses']].to_string(index=False))
        
        print("\n=== PREFERENCE DISTRIBUTION (COMBINED) ===")
        pref_cols = ['category', 'preference_base_pct', 'preference_equal_pct', 'preference_provoked_pct']
        print(combined_df[pref_cols].to_string(index=False))
        
        print("\n=== CORRECTNESS DISTRIBUTION (COMBINED) ===")
        corr_cols = ['category', 'correctness_base_pct', 'correctness_equal_pct', 'correctness_provoked_pct']
        print(combined_df[corr_cols].to_string(index=False))
        
        print("\n=== MANIPULATION DISTRIBUTION (COMBINED) ===")
        manip_cols = ['category', 'manipulation_base_pct', 'manipulation_equal_pct', 'manipulation_provoked_pct']
        print(combined_df[manip_cols].to_string(index=False))
    
    return viz_ready_df



colors = {
    'base': "#7a0177",      # Purple
    'equal': '#db3657',     # Orange
    'provoked': '#ffa600'   # Blue
}

def create_single_judge_compact_visualization(enhanced_df, 
                                             font_scale=1.0, 
                                             figsize=(12, 8),
                                             bar_height=0.7,
                                             judge_label='Combined Judges',
                                             show_legend=True):
    """
    Create a compact visualization showing all three metrics for a single judge.
    Metrics are displayed as three columns with categories as rows.
    
    Parameters:
    enhanced_df (pd.DataFrame): Enhanced results DataFrame for a single judge or combined results
    font_scale (float): Scaling factor for all font sizes (default: 1.0)
    figsize (tuple): Figure size (width, height) in inches
    bar_height (float): Height of bars as fraction of available space (0.0-1.0)
    judge_label (str): Label for the judge being displayed
    show_legend (bool): Whether to show legend
    
    Returns:
    tuple: (fig, axes) matplotlib figure and axes objects
    """
    

    
    # Create lookup dictionary
    data_lookup = {row['category']: row for _, row in enhanced_df.iterrows()}
    
    # Group and sort categories
    grouped_categories = {}
    for category, group in category_groups.items():
        if group not in grouped_categories:
            grouped_categories[group] = []
        grouped_categories[group].append(category)
    
    # Order groups and categories
    group_order = ['Skewed Presentation', 'Misleading Claims', 'Emotional Pressure']
    ordered_categories = []
    for group in group_order:
        if group in grouped_categories:
            sorted_cats = sorted(grouped_categories[group])
            ordered_categories.extend(sorted_cats)
    
    # Filter categories that exist in the dataset
    valid_categories = []
    category_labels = []
    
    for category in ordered_categories:
        if category in data_lookup:
            valid_categories.append(category)
            category_labels.append(category_display_names.get(category, category))
    
    n_categories = len(valid_categories)
    
    # Create figure with 3 subplots (one for each metric)
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
    
    # Define metrics and their column mappings
    metrics = [
        ('Preferred', 'base_preferred_pct', 'equal_preferred_pct', 'provoked_preferred_pct'),
        ('Correctness', 'correctness_base_higher_pct', 'correctness_equal_pct', 'correctness_base_lower_pct'),
        ('Manipulation', 'metric_base_higher_pct', 'metric_equal_pct', 'metric_base_lower_pct')
    ]
    
    # Track group boundaries for separators
    group_boundaries = []
    current_pos = 0
    for group in group_order:
        if group in grouped_categories:
            group_size = len([cat for cat in grouped_categories[group] if cat in data_lookup])
            current_pos += group_size
            if current_pos < n_categories:
                group_boundaries.append(n_categories - current_pos)
    
    # Create each metric subplot
    for ax_idx, (metric_name, base_col, equal_col, provoked_col) in enumerate(metrics):
        ax = axes[ax_idx]
        
        # Set up the subplot
        ax.set_xlim(0, 1)
        ax.set_ylim(0, n_categories)
        
        # Draw bars for each category
        for cat_idx, category in enumerate(valid_categories):
            cat_data = data_lookup[category]
            
            # Y position (flip to have first category at top)
            y_pos = n_categories - cat_idx - 1
            y_center = y_pos + 0.5
            y_bottom = y_center - bar_height/2
            
            # Get percentages
            base_pct = cat_data[base_col]
            equal_pct = cat_data[equal_col]
            provoked_pct = cat_data[provoked_col]
            
            # Normalize to ensure they sum to 100
            total = base_pct + equal_pct + provoked_pct
            if total > 0:
                base_pct = (base_pct / total) * 100
                equal_pct = (equal_pct / total) * 100
                provoked_pct = (provoked_pct / total) * 100
            
            # Draw horizontal stacked bar
            # Base segment (left)
            if base_pct > 0:
                rect = patches.Rectangle((0, y_bottom), base_pct/100, bar_height,
                                        facecolor=colors['base'], 
                                        edgecolor='white', linewidth=0.5)
                ax.add_patch(rect)
            
            # Equal segment (middle)
            if equal_pct > 0:
                rect = patches.Rectangle((base_pct/100, y_bottom), equal_pct/100, bar_height,
                                        facecolor=colors['equal'], 
                                        edgecolor='white', linewidth=0.5)
                ax.add_patch(rect)
            
            # Provoked segment (right)
            if provoked_pct > 0:
                rect = patches.Rectangle(((base_pct + equal_pct)/100, y_bottom), provoked_pct/100, bar_height,
                                        facecolor=colors['provoked'], 
                                        edgecolor='white', linewidth=0.5)
                ax.add_patch(rect)
        
        # Add group separators
        for boundary in group_boundaries:
            ax.axhline(y=boundary, color='white', linewidth=2)
        
        # Add subtle grid
        for i in range(n_categories + 1):
            ax.axhline(y=i, color='gray', linewidth=0.3, alpha=0.2)
        
        # Set subplot title (metric name)
        ax.set_title(metric_name, fontsize=11 * font_scale, fontweight='bold', pad=10)
        
        # X-axis configuration
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xticklabels(['0%', '50%', '100%'], fontsize=9 * font_scale)
        
        # Remove spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(0.5)
        ax.spines['left'].set_visible(False) if ax_idx > 0 else ax.spines['left'].set_linewidth(0.5)
        
        # Remove tick marks
        ax.tick_params(axis='both', which='both', length=0)
    
    # Set y-axis labels only on the leftmost subplot
    axes[0].set_yticks([i + 0.5 for i in range(n_categories)])
    axes[0].set_yticklabels(category_labels[::-1], fontsize=9 * font_scale)
    
    # Add group labels on the right side of the rightmost subplot
    ax_right = axes[2]
    current_pos = 0
    for group in group_order:
        if group in grouped_categories:
            group_size = len([cat for cat in grouped_categories[group] if cat in data_lookup])
            if group_size > 0:
                group_center = n_categories - current_pos - group_size/2
                ax_right.text(1.05, group_center, group.replace(' ', '\n'),
                            rotation=270, va='center', ha='left',
                            fontsize=9 * font_scale, fontweight='bold',
                            color='#555555')
                current_pos += group_size
    
    # Overall title
    # y_title = 0.98
    y_title = 0.94
    fig.suptitle(f'{judge_label} - Evaluation Results', 
                fontsize=13 * font_scale, fontweight='bold', y=y_title)
    
    # Add legend at the bottom
    if show_legend:
        legend_elements = [
            patches.Patch(facecolor=colors['base'], label='Base Chosen'),
            patches.Patch(facecolor=colors['equal'], label='Tie'),
            patches.Patch(facecolor=colors['provoked'], label='Provoked Chosen')
        ]
        fig.legend(handles=legend_elements, loc='lower center', 
                  bbox_to_anchor=(0.5, -0.05), ncol=3, 
                  fontsize=9 * font_scale, frameon=False)
    
    # Adjust layout
    plt.subplots_adjust(wspace=0.08, hspace=0.1)
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    
    return fig, axes


def visualize_single_judge_results(combined_df, judge_label='Combined Judges', **kwargs):
    """
    Wrapper function to easily visualize single judge results.
    
    Parameters:
    combined_df (pd.DataFrame): Results DataFrame (either single judge or combined)
    judge_label (str): Label for the judge
    **kwargs: Additional arguments to pass to create_single_judge_compact_visualization
    
    Returns:
    tuple: (fig, axes) matplotlib figure and axes
    """
    
    default_params = {
        'font_scale': 1.2,
        'figsize': (12, 8),
        'bar_height': 0.7,
        'show_legend': True
    }
    
    # Update defaults with any provided kwargs
    default_params.update(kwargs)
    
    fig, axes = create_single_judge_compact_visualization(
        combined_df,
        judge_label=judge_label,
        **default_params
    )
    
    return fig, axes


