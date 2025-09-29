"""
Enhanced Multi-Layer Steering Vector Analysis

This script trains steering vectors for all layers of a model and evaluates them
on the full dataset. Key improvements:
- Stores complete preference score data for all metrics
- Flexible plotting system for any metric from preference scores
- Better data organization and access methods
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle
import os
from pathlib import Path

import scipy
import nltk

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
import math

from notebooks.notebook_functions import get_judge_result_file
from steering_vec_functions.datasets import format_question
from steering_vec_functions.model_utils import get_model_and_tokenizer
from steering_vec_functions.generate_provoked_and_steered_responses import create_steering_vector, SteeringVector
from tqdm import tqdm


def get_prompts_responses(responses, tokenizer):
    """Extract prompts and responses from the dataset."""
    prompts_base = []
    prompts_prov = []
    responses_base = []
    responses_prov = []

    for entry in responses:
        prompts_base.append(format_question(entry['base_prompt'], tokenizer))
        prompts_prov.append(format_question(entry['suggestive_prompt'], tokenizer))
        responses_base.append(entry['base_response'])
        responses_prov.append(entry['suggestive_response'])

    return prompts_base, prompts_prov, responses_base, responses_prov


def preference_score_steered(steering_vector: SteeringVector, prompt: str, resp_a: str, resp_b: str) -> dict:
    """Calculate preference scores between two responses."""
    norm_log_llh_pertok_a, response_length_a = steering_vector.get_steered_log_likelihood(prompt, resp_a)
    norm_log_llh_pertok_b, response_length_b = steering_vector.get_steered_log_likelihood(prompt, resp_b)

    s_a = norm_log_llh_pertok_a
    s_b = norm_log_llh_pertok_b

    # Softmax over scores
    p_a = math.exp(s_a) / (math.exp(s_a) + math.exp(s_b))

    return {
        "prob_a": p_a,
        "prob_b": 1 - p_a,
        "resp_a": {
            "neg_log_likelihood": -s_a,
            "response_length": response_length_a
        },
        "resp_b": {
            "neg_log_likelihood": -s_b,
            "response_length": response_length_b
        }
    }


def get_num_layers(model):
    """Detect the number of layers in the model."""
    if hasattr(model, 'config'):
        if hasattr(model.config, 'num_hidden_layers'):
            return model.config.num_hidden_layers
        elif hasattr(model.config, 'n_layer'):
            return model.config.n_layer
    
    # Fallback: try to count layers manually
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return len(model.model.layers)
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return len(model.transformer.h)
    
    raise ValueError("Could not determine number of layers in model")


def extract_metric_from_preference_scores(preference_scores_list, metric_path):
    """
    Extract a specific metric from a list of preference score dictionaries.
    
    Args:
        preference_scores_list: List of preference score dicts for one layer
        metric_path: String like "prob_a", "resp_a.neg_log_likelihood", etc.
    
    Returns:
        List of values for the specified metric
    """
    values = []
    for pref_scores in preference_scores_list:
        if pref_scores is None:
            values.append(None)
            continue
            
        # Navigate nested dict using dot notation
        try:
            value = pref_scores
            for key in metric_path.split('.'):
                value = value[key]
            values.append(value)
        except (KeyError, TypeError):
            values.append(None)
    
    return values


def get_available_metrics():
    """Return list of available metric paths for plotting."""
    return [
        "prob_a",
        "prob_b", 
        "resp_a.neg_log_likelihood",
        "resp_b.neg_log_likelihood",
        "resp_a.response_length",
        "resp_b.response_length",
        "resp_a.perplexity",
        "resp_b.perplexity"
    ]


def train_steering_vectors_all_layers(
    model,
    tokenizer,
    responses,
    prompts_base,
    prompts_prov,
    responses_base,
    responses_prov,
    save_folder="./results/steering_vectors_multilayer",
    **steering_params
):
    """Train steering vectors for all layers and save results."""
    
    # Create save folder if it doesn't exist
    Path(save_folder).mkdir(parents=True, exist_ok=True)
    
    # Get number of layers
    num_layers = get_num_layers(model)
    num_samples = len(prompts_base)
    print(f"Model has {num_layers} layers")
    print(f"Evaluating on {num_samples} samples")
    
    # Initialize results dictionary - now stores complete preference score data
    results = {
        "layers": [],
        "norms": [],
        "preference_scores": [],  # List of lists: [layer_idx][sample_idx] = full_pref_scores_dict
        "model_name": steering_params.get("model_name", "unknown"),
        "num_samples": num_samples,
        "available_metrics": get_available_metrics()  # For reference
    }
    
    # Try to load existing results if the process was interrupted
    results_path = os.path.join(save_folder, "results.pkl")
    if os.path.exists(results_path):
        print("Found existing results file, loading to resume...")
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
        start_layer = len(results["layers"])
        print(f"Resuming from layer {start_layer}")
    else:
        start_layer = 0
    
    # Train steering vector for each layer
    for layer_idx in tqdm(range(start_layer, num_layers), desc="Training layers"):
        print(f"\n{'='*50}")
        print(f"Training layer {layer_idx}/{num_layers-1}")
        print(f"{'='*50}")
        
        try:
            # Create steering vector for this layer
            steering_vector = create_steering_vector(
                model,
                tokenizer,
                layer_idx,
                dataset=responses,
                **steering_params
            )
            
            # Get norm
            norm = steering_vector.vector.norm().item()
            
            # Save steering vector
            vector_filename = f"steering_vector_layer_{layer_idx}.pt"
            vector_path = os.path.join(save_folder, vector_filename)
            torch.save(steering_vector.vector, vector_path)
            
            # Calculate preference scores for all samples - store complete data
            layer_pref_scores = []
            
            for sample_idx in tqdm(range(num_samples), desc=f"Evaluating samples", leave=False):
                prompt = prompts_prov[sample_idx]
                resp_a = responses_base[sample_idx]
                resp_b = responses_prov[sample_idx]
                
                # Calculate preference - store the complete dictionary
                pref_scores = preference_score_steered(
                    steering_vector, prompt, resp_a, resp_b
                )
                
                layer_pref_scores.append(pref_scores)
            
            # Store results
            results["layers"].append(layer_idx)
            results["norms"].append(norm)
            results["preference_scores"].append(layer_pref_scores)
            
            # Calculate mean preference for logging (backward compatibility)
            prob_a_values = extract_metric_from_preference_scores(layer_pref_scores, "prob_a")
            prob_a_values = [v for v in prob_a_values if v is not None]
            if prob_a_values:
                mean_p_a = np.mean(prob_a_values)
                std_p_a = np.std(prob_a_values)
                print(f"Layer {layer_idx}: Norm={norm:.4f}, Mean P_a={mean_p_a:.3f} (±{std_p_a:.3f})")
            
            # Save results after each layer (incremental saving)
            with open(results_path, 'wb') as f:
                pickle.dump(results, f)
            print(f"Saved results up to layer {layer_idx}")
            
        except Exception as e:
            print(f"Error at layer {layer_idx}: {e}")
            # Store None for failed layers
            results["layers"].append(layer_idx)
            results["norms"].append(None)
            results["preference_scores"].append([None] * num_samples)
            
            # Save even failed results
            with open(results_path, 'wb') as f:
                pickle.dump(results, f)
    
    print(f"\nAll layers complete! Results saved to {results_path}")
    return results


def load_results(save_folder="./results/steering_vectors_multilayer"):
    """Load saved results from disk."""
    results_path = os.path.join(save_folder, "results.pkl")
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    return results


def plot_results(results, save_plots=True, save_folder="./results/steering_vectors_multilayer"):
    """Plot norms and preference scores over layers."""
    
    # Filter out None values and calculate means
    valid_indices = [i for i, n in enumerate(results["norms"]) if n is not None]
    layers = [results["layers"][i] for i in valid_indices]
    norms = [results["norms"][i] for i in valid_indices]
    
    # Calculate mean and std for preference scores using new extraction method
    p_a_means = []
    p_a_stds = []
    for i in valid_indices:
        if results["preference_scores"][i] is not None:
            prob_a_values = extract_metric_from_preference_scores(
                results["preference_scores"][i], "prob_a"
            )
            scores = [s for s in prob_a_values if s is not None]
            if scores:
                p_a_means.append(np.mean(scores))
                p_a_stds.append(np.std(scores))
            else:
                p_a_means.append(None)
                p_a_stds.append(None)
        else:
            p_a_means.append(None)
            p_a_stds.append(None)
    
    # Filter out any remaining None values
    valid_p_a = [(l, m, s) for l, m, s in zip(layers, p_a_means, p_a_stds) if m is not None]
    if valid_p_a:
        layers_p_a, p_a_means, p_a_stds = zip(*valid_p_a)
    else:
        layers_p_a, p_a_means, p_a_stds = [], [], []
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Norms over layers
    axes[0].plot(layers, norms, 'b-o', markersize=4)
    axes[0].set_xlabel('Layer Index')
    axes[0].set_ylabel('Steering Vector Norm')
    axes[0].set_title('Steering Vector Norm vs Layer')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Preference score over layers with error bars
    axes[1].errorbar(layers_p_a, p_a_means, yerr=p_a_stds, 
                     fmt='r-s', markersize=4, capsize=3, alpha=0.7)
    axes[1].set_xlabel('Layer Index')
    axes[1].set_ylabel('P(A) - Preference for Base Response')
    axes[1].set_title(f'Preference Score vs Layer (n={results.get("num_samples", "?")} samples)')
    axes[1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    axes[1].grid(True, alpha=0.3)
    # axes[1].set_ylim([0, 1])
    
    # Add shaded region for standard deviation
    if p_a_means and p_a_stds:
        axes[1].fill_between(layers_p_a, 
                            [m - s for m, s in zip(p_a_means, p_a_stds)],
                            [m + s for m, s in zip(p_a_means, p_a_stds)],
                            alpha=0.2, color='red')
    
    plt.suptitle(f'Steering Vector Analysis - {results.get("model_name", "Model")}')
    plt.tight_layout()
    
    if save_plots:
        plot_path = os.path.join(save_folder, "analysis_plots.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Plots saved to {plot_path}")
    
    plt.show()
    
    return fig


def plot_single_metric(
    results,
    metric="prob_a",
    metric_b=None,
    save_plot=False,
    save_folder="./results/steering_vectors_multilayer",
    show_std=True,
    fontsize_scaler=1.0,
    figsize=(8, 5), 
    title = None,
    return_fig = False,
    y_label = None,
    color = None,
    line_label = None,
):
    """
    Plot one or two metrics over layers.

    Args:
        results: Results dictionary
        metric: First metric to plot (string)
        metric_b: Optional second metric to plot (string)
        save_plot: Whether to save the plot
        save_folder: Folder to save plot in
        show_std: Whether to show standard deviation (for preference metrics only)
        fontsize_scaler: Multiplier for font sizes in the plot
    """

    fig = plt.figure(figsize=figsize)
    linewidth=2.4

    def get_layer_stats(metric_name):
        if metric_name == "norms":
            valid_indices = [i for i, d in enumerate(results["norms"]) if d is not None]
            layers = [results["layers"][i] for i in valid_indices]
            means = [results["norms"][i] for i in valid_indices]
            stds = [0 for _ in means]
        else:
            if metric_name not in get_available_metrics():
                raise ValueError(f"Metric '{metric_name}' not available. Choose from: {get_available_metrics()}")
            valid_indices = [i for i, pref_scores in enumerate(results["preference_scores"])
                             if pref_scores is not None and len(pref_scores) > 0]
            layers = [results["layers"][i] for i in valid_indices]
            means = []
            stds = []
            for i in valid_indices:
                values = extract_metric_from_preference_scores(
                    results["preference_scores"][i], metric_name
                )
                values = [v for v in values if v is not None]
                if values:
                    means.append(np.mean(values))
                    stds.append(np.std(values))
        return layers, means, stds

    # Plot first metric
    layers, means, stds = get_layer_stats(metric)
    if color is None:
        color = 'red' if 'prob' in metric else 'green' if 'neg_log_likelihood' in metric else 'blue'

    if line_label is not None:
        label = line_label[metric]
    else:
        label = metric.replace(".", " ").title()
    if show_std and stds and any(stds):
        plt.errorbar(layers, means, yerr=stds, 
                    #  fmt=f'{color[0]}-s', 
                    fmt=color,
                    marker="s", linestyle="-",
                     markersize=5, capsize=3, alpha=0.7, label=label)
        plt.fill_between(layers,
                         [m - s for m, s in zip(means, stds)],
                         [m + s for m, s in zip(means, stds)],
                         alpha=0.2, color=color)
    else:
        plt.plot(layers, means, f'{color}', marker="s", linestyle="-", markersize=5, label=label, linewidth=linewidth)

    # Plot second metric if provided
    if metric_b:
        layers_b, means_b, stds_b = get_layer_stats(metric_b)
        # color_b = 'orange' if 'prob' in metric_b else 'purple' if 'neg_log_likelihood' in metric_b else 'cyan'
        # color_b = "purple"
        color_b = "#ffa600"
        if line_label is not None:
            label_b = line_label[metric_b]
        else:
            label_b = metric_b.replace(".", " ").title()
        if show_std and stds_b and any(stds_b):
            plt.errorbar(layers_b, means_b, yerr=stds_b, fmt=f'{color_b}', marker="s", linestyle="-", markersize=5, capsize=3, alpha=0.7, label=label_b)
            plt.fill_between(layers_b,
                             [m - s for m, s in zip(means_b, stds_b)],
                             [m + s for m, s in zip(means_b, stds_b)],
                             alpha=0.2, color=color_b)
        else:
            plt.plot(layers_b, means_b, f'{color_b}',marker="s", linestyle="-", markersize=5, label=label_b, linewidth=linewidth)

    # Set y-label and title
    if metric_b:
        if y_label:
            plt.ylabel(y_label, fontsize=12 * fontsize_scaler)
        else:
            plt.ylabel('Metric Value', fontsize=12 * fontsize_scaler)
        if not title:
            title = f'{label} & {label_b} vs Layer (n={results.get("num_samples", "?")} samples)'
        plt.title(title, fontsize=14 * fontsize_scaler)
    else:
        if y_label:
            plt.ylabel(y_label, fontsize=12 * fontsize_scaler)
        else:
            if 'prob' in metric:
                plt.ylabel(f'{metric.upper()} - Preference Probability', fontsize=12 * fontsize_scaler)
            elif 'neg_log_likelihood' in metric:
                plt.ylabel(f'{metric.replace(".", " ")} (Negative Log-Likelihood)', fontsize=12 * fontsize_scaler)
            elif 'response_length' in metric:
                plt.ylabel(f'{metric.replace(".", " ")} (Response Length)', fontsize=12 * fontsize_scaler)
        
        if 'prob' in metric and metric == "prob_a":
            plt.axhline(y=0.5, color='gray', linestyle='--', alpha=1.0)
            # plt.ylim([0, 1])
        
        if not title:
            title = f'{label} vs Layer (n={results.get("num_samples", "?")} samples)'
        plt.title(title, fontsize=13 * fontsize_scaler)

    plt.xlabel('Layer Index', fontsize=12 * fontsize_scaler)
    plt.grid(True, alpha=0.3)
    if metric_b:
        plt.legend(fontsize=11 * fontsize_scaler)

    plt.xticks(fontsize=10 * fontsize_scaler)
    plt.yticks(fontsize=10 * fontsize_scaler)

    if save_plot:
        if metric_b:
            filename = f"{metric.replace('.', '_')}_and_{metric_b.replace('.', '_')}_plot.png"
        else:
            filename = f"{metric.replace('.', '_')}_plot.png"
        plot_path = os.path.join(save_folder, filename)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {plot_path}")

    plt.show()
    if return_fig:
        return fig


def get_layer_statistics(results, layer_idx):
    """Get detailed statistics for a specific layer."""
    
    if layer_idx not in results["layers"]:
        raise ValueError(f"Layer {layer_idx} not found in results")
    
    idx = results["layers"].index(layer_idx)
    
    stats = {
        "layer": layer_idx,
        "norm": results["norms"][idx]
    }
    
    # Get statistics for all available metrics
    if results["preference_scores"][idx] is not None:
        for metric in get_available_metrics():
            values = extract_metric_from_preference_scores(
                results["preference_scores"][idx], metric
            )
            values = [v for v in values if v is not None]
            if values:
                stats[f"{metric}_mean"] = np.mean(values)
                stats[f"{metric}_std"] = np.std(values)
                stats[f"{metric}_min"] = np.min(values)
                stats[f"{metric}_max"] = np.max(values)
                stats[f"{metric}_median"] = np.median(values)
        
        # Count valid samples
        stats["num_samples"] = len([p for p in results["preference_scores"][idx] if p is not None])
    
    return stats


def print_summary_statistics(results):
    """Print summary statistics for all layers."""
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    # Find best layers
    valid_norms = [(l, n) for l, n in zip(results["layers"], results["norms"]) if n is not None]
    
    if valid_norms:
        best_norm_layer = max(valid_norms, key=lambda x: x[1])
        print(f"\nHighest norm: Layer {best_norm_layer[0]} (norm={best_norm_layer[1]:.4f})")
    
    # Find layer with highest mean preference score
    best_p_a = -1
    best_p_a_layer = None
    
    for i, layer in enumerate(results["layers"]):
        if results["preference_scores"][i] is not None:
            prob_a_values = extract_metric_from_preference_scores(
                results["preference_scores"][i], "prob_a"
            )
            scores = [s for s in prob_a_values if s is not None]
            if scores:
                mean_score = np.mean(scores)
                if mean_score > best_p_a:
                    best_p_a = mean_score
                    best_p_a_layer = layer
    
    if best_p_a_layer is not None:
        print(f"Highest mean P(A): Layer {best_p_a_layer} (mean={best_p_a:.4f})")
    
    # Print per-layer summary
    print("\n" + "-"*80)
    print(f"{'Layer':<8} {'Norm':<10} {'Mean P(A)':<12} {'Std P(A)':<10} {'Mean NLL A':<12} {'Mean NLL B':<12}")
    print("-"*80)
    
    for i, layer in enumerate(results["layers"]):
        norm = results["norms"][i]
        
        if norm is not None and results["preference_scores"][i] is not None:
            # Extract various metrics
            prob_a_values = extract_metric_from_preference_scores(
                results["preference_scores"][i], "prob_a"
            )
            nll_a_values = extract_metric_from_preference_scores(
                results["preference_scores"][i], "resp_a.neg_log_likelihood"
            )
            nll_b_values = extract_metric_from_preference_scores(
                results["preference_scores"][i], "resp_b.neg_log_likelihood"
            )
            
            prob_a_clean = [v for v in prob_a_values if v is not None]
            nll_a_clean = [v for v in nll_a_values if v is not None]
            nll_b_clean = [v for v in nll_b_values if v is not None]
            
            if prob_a_clean:
                mean_p_a = np.mean(prob_a_clean)
                std_p_a = np.std(prob_a_clean)
                mean_nll_a = np.mean(nll_a_clean) if nll_a_clean else float('nan')
                mean_nll_b = np.mean(nll_b_clean) if nll_b_clean else float('nan')
                
                print(f"{layer:<8} {norm:<10.4f} {mean_p_a:<12.4f} {std_p_a:<10.4f} {mean_nll_a:<12.4f} {mean_nll_b:<12.4f}")
            else:
                print(f"{layer:<8} {norm:<10.4f} {'N/A':<12} {'N/A':<10} {'N/A':<12} {'N/A':<12}")
        elif norm is not None:
            print(f"{layer:<8} {norm:<10.4f} {'N/A':<12} {'N/A':<10} {'N/A':<12} {'N/A':<12}")
        else:
            print(f"{layer:<8} {'Failed':<10} {'Failed':<12} {'Failed':<10} {'Failed':<12} {'Failed':<12}")
    
    print("="*80 + "\n")
    
    # Print available metrics for reference
    print("Available metrics for plotting:")
    for metric in get_available_metrics():
        print(f"  - {metric}")
    print()


# Main execution
if __name__ == "__main__":
    # Load data
    data, responses = get_judge_result_file("manipulation", "GPT4Base")
    
    # Load model and tokenizer
    model_name = "google/gemma-2-2b-it"
    model, tokenizer = get_model_and_tokenizer(model_name, use_quantizer=False, low_memory_load=True)
    print(f"Loaded model: {model_name}")
    
    # Get prompts and responses
    prompts_base, prompts_prov, responses_base, responses_prov = get_prompts_responses(responses, tokenizer=tokenizer)
    
    # Steering parameters
    # steering_params = {
    #     "num_iters": 20,
    #     "lr": 0.01,
    #     "generation_length": 100,
    #     "generation_length_optimization": 100,
    #     "temperature": 0.7,
    #     "model_name": model_name,
    #     "use_load_vector": False,
    #     "max_norm": 10,
    #     "multi_sample": False,
    #     "use_clamp_steer": False,
    #     "steer_vector_name": ""
    # }
    steering_params = {
        "num_iters": 30,
        "lr": 0.1,
        "generation_length": 200,
        "generation_length_optimization": 50,
        "temperature": 0.7,
        "model_name": model_name,
        "use_load_vector": False,
        "max_norm": None,
        "multi_sample": False,
        "use_clamp_steer": False,
        "steer_vector_name": ""
    }
    
    # Train steering vectors for all layers
    results = train_steering_vectors_all_layers(
        model,
        tokenizer,
        responses,
        prompts_base,
        prompts_prov,
        responses_base,
        responses_prov,
        save_folder="./results/steering_vectors_multilayer2",
        **steering_params
    )
    
    # Print summary statistics
    print_summary_statistics(results)
    
    # Plot results
    plot_results(results, save_plots=True)
    
    # Example usage of new plotting capabilities
    print("\nExample plots for different metrics:")
    plot_single_metric(results, metric="prob_a", show_std=True)
    plot_single_metric(results, metric="resp_a.neg_log_likelihood", show_std=True)
    plot_single_metric(results, metric="resp_b.neg_log_likelihood", show_std=True)
    
    # Example: Load and plot saved results later
    # This also works if training was interrupted - partial results are saved!
    # loaded_results = load_results("./results/steering_vectors_multilayer")
    # print_summary_statistics(loaded_results)
    # plot_results(loaded_results)
    # plot_single_metric(loaded_results, metric="prob_b")
    # plot_single_metric(loaded_results, metric="resp_a.response_length")