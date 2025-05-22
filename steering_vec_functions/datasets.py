import random
import requests

import os
import json
import random
# Define a shorthand type for model-written eval datum
MWEData = list[dict[str, str]]


def format_question(question: str, tokenizer=None, system_prompt="You are a helpful assistant.") -> str:
    """
    Formats a single question using the tokenizer's chat template, ready for model inference.
    
    Args:
        question: The user's question to format
        tokenizer: HuggingFace tokenizer instance (will load default if None)
        system_prompt: Optional system prompt to include
        
    Returns:
        Formatted prompt string ready for model input
    """
    
    # Create messages with just system and user, no assistant response yet
    # check if tokenizer is of type gemma
    tokenizer_name = tokenizer.__class__.__name__.lower()
    # print(f"Tokenizer name: {tokenizer_name}")
    system_name = "system"
    if "gemma" in tokenizer_name:
        system_name = "model"
        messages = [
            {"role": "user", "content": question}
        ]

    else:
        messages = [
            {"role": system_name, "content": system_prompt},
            {"role": "user", "content": question}
        ]
    
    # Apply tokenizer's chat template with add_generation_prompt=True to include the assistant prefix
    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False,
        add_generation_prompt=True  # This adds the assistant prefix token(s)
    )
    
    return formatted_prompt



def get_sycophancy_dataset(data_path="./data/", data_type="feedback"):
    """
    Ensures the specified sycophancy dataset exists in the given data path.
    Downloads it if it does not exist and returns the JSON dataset.
    
    Parameters:
        data_path (str): Path to the directory where the dataset should be stored.
        data_type (str): Type of dataset to load. Options are "feedback", "answer", or "are_you_sure".
    
    Returns:
        list: The JSON dataset loaded from the file.
    """
    # Define dataset URLs and file names
    dataset_info = {
        "feedback": "https://raw.githubusercontent.com/meg-tong/sycophancy-eval/main/datasets/feedback.jsonl",
        "answer": "https://raw.githubusercontent.com/meg-tong/sycophancy-eval/main/datasets/answer.jsonl",
        "are_you_sure": "https://raw.githubusercontent.com/meg-tong/sycophancy-eval/main/datasets/are_you_sure.jsonl"
    }

    if data_type not in dataset_info:
        raise ValueError(f"Invalid data_type '{data_type}'. Must be one of {list(dataset_info.keys())}.")

    dataset_url = dataset_info[data_type]
    folder_path = os.path.join(data_path, f"sycophancy_{data_type}")
    file_name = f"{data_type}.jsonl"
    file_path = os.path.join(folder_path, file_name)

    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)

    # Check if the file already exists
    if not os.path.exists(file_path):
        print(f"Downloading {data_type} dataset from {dataset_url}...")
        response = requests.get(dataset_url)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f"{data_type.capitalize()} dataset downloaded and saved to {file_path}")
        else:
            print(f"Failed to download {data_type} dataset. HTTP Status Code: {response.status_code}")
            return None
    else:
        print(f"{data_type.capitalize()} dataset already exists at {file_path}")

    # Load and return the JSON dataset
    with open(file_path, 'r') as f:
        dataset = [json.loads(line) for line in f]
    return dataset