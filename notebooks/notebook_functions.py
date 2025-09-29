
import json


def load_data_json(file_path):
    """Load JSON data from a file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def get_judge_result_file(dataset, model_name,  snelius_prefix = "results/final_judge_results_v2/"):
    """
    Return the loaded judge result data and responses based on dataset and model name.

    Parameters:
        dataset (str): 'manipulation' or 'sycophantic'
        model_name (str): e.g. 'Claude', 'GPT4Nano', 'GPT4Mini', 'GPT4Base', 'ClaudeHaiku'

    Returns:
        tuple: (data, responses)
    """

    if dataset == "manipulation":
        if model_name.lower() == "claude":
            file_path = snelius_prefix + "judged_responses_manipulation_Claude_FINAL.json"
        elif model_name.lower() == "gpt4base":
            file_path = snelius_prefix + "judged_responses_manipulation_GPT4Base_FINAL.json"
        elif model_name.lower() == "gpt4nano":
            file_path = snelius_prefix + "judged_responses_manipulation_GPT4Nano_FINAL.json"
        elif model_name.lower() == "gpt4mini":
            file_path = snelius_prefix + "judged_responses_manipulation_GPT4Mini_FINAL.json"
        else:
            raise ValueError(f"Unknown model name for manipulative: {model_name}")
    elif dataset == "sycophantic":
        if model_name.lower() == "gpt4base":
            file_path = snelius_prefix + "judged_responses_sycophancy_GPT4Base.json"
        elif model_name.lower() == "gpt4mini":
            file_path = snelius_prefix + "judged_responses_sycophancy_GPT4Mini.json"
        elif model_name.lower() == "gpt4nano":
            file_path = snelius_prefix + "judged_responses_sycophancy_GPT4Nano.json"
        elif model_name.lower() == "claude":
            file_path = snelius_prefix + "judged_responses_sycophancy_ClaudeHaiku.json"
        else:
            raise ValueError(f"Unknown model name for sycophantic: {model_name}")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    data = load_data_json(file_path)
    responses = data['responses']
    return data, responses

