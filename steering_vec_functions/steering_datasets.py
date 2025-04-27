import json
import random



import os
import json
import random
# Define a shorthand type for model-written eval datum
MWEData = list[dict[str, str]]


def download_caa_data(data_path: str):
    """Download the CAA dataset if it does not exist."""
    os.makedirs(f"{data_path}/sycophancy", exist_ok=True)
    train_url = "https://raw.githubusercontent.com/nrimsky/CAA/main/datasets/generate/sycophancy/generate_dataset.json"
    test_url = "https://raw.githubusercontent.com/nrimsky/CAA/main/datasets/test/sycophancy/test_dataset_ab.json"
    train_file = f"{data_path}/sycophancy/generate_dataset.json"
    test_file = f"{data_path}/sycophancy/test_dataset_ab.json"

    if not os.path.exists(train_file):
        os.system(f"wget -P {data_path}/sycophancy {train_url}")
    if not os.path.exists(test_file):
        os.system(f"wget -P {data_path}/sycophancy {test_url}")

def load_caa_dataset(data_path: str, behavior_type: str = "sycophancy"):
    """
    Load the CAA dataset for the specified behavior type.
    If the dataset does not exist, it will be downloaded.

    Args:
        data_path (str): Path to the data directory.
        behavior_type (str): Type of behavior dataset to load (default: "sycophancy").

    Returns:
        tuple: train_data and test_data as lists of dictionaries.
    """
    # Ensure the dataset is downloaded
    download_caa_data(data_path)

    # Load the train and test datasets
    train_file = f"{data_path}/{behavior_type}/generate_dataset.json"
    test_file = f"{data_path}/{behavior_type}/test_dataset_ab.json"

    with open(train_file, "r") as f:
        train_data = json.load(f)

    with open(test_file, "r") as f:
        test_data = json.load(f)

    # Shuffle the datasets for randomness
    random.seed(42)
    random.shuffle(train_data)
    random.shuffle(test_data)

    return train_data, test_data



def make_pos_neg_pair(mwe_data: MWEData, tokenizer) -> tuple[str, str]:
    """Creates a (positive, negative) pair for getting contrastive activations using tokenizer formatting"""

    # Use tokenizer's chat template with system prompt
    system_prompt = "You are a helpful, honest assistant."
    
    # For positive example
    pos_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": mwe_data["question"]},
        # {"role": "assistant", "content": mwe_data["answer_matching_behavior"]}
    ]
    
    # For negative example
    neg_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": mwe_data["question"]},
        # {"role": "assistant", "content": mwe_data["answer_not_matching_behavior"]}
    ]
    
    # Apply tokenizer's chat template
    pos = tokenizer.apply_chat_template(pos_messages, tokenize=False, add_generation_prompt=True)
    neg = tokenizer.apply_chat_template(neg_messages, tokenize=False, add_generation_prompt=True)

    pos += mwe_data["answer_matching_behavior"]
    neg += mwe_data["answer_not_matching_behavior"]
    
    return pos, neg


def format_caa_dataset(list_mwe_data: list[MWEData], tokenizer) -> list[tuple[str, str]]:
    """Creates a list of (positive, negative) pairs for getting contrastive activations"""
  
    return [make_pos_neg_pair(mwe_data, tokenizer) for mwe_data in list_mwe_data]



from transformers import AutoTokenizer

def format_question(question: str, tokenizer=None, system_prompt="You are a helpful, honest assistant.") -> str:
    """
    Formats a single question using the tokenizer's chat template, ready for model inference.
    
    Args:
        question: The user's question to format
        tokenizer: HuggingFace tokenizer instance (will load default if None)
        system_prompt: Optional system prompt to include
        
    Returns:
        Formatted prompt string ready for model input
    """
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")  # replace with your model
    
    # Create messages with just system and user, no assistant response yet
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    
    # Apply tokenizer's chat template with add_generation_prompt=True to include the assistant prefix
    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False,
        add_generation_prompt=True  # This adds the assistant prefix token(s)
    )
    
    return formatted_prompt