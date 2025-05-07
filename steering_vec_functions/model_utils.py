import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from functools import partial
from typing import Tuple

def get_model_and_tokenizer(model_name: str, use_quantizer=False, low_memory_load=False) -> Tuple:
    """Load model and tokenizer with appropriate configuration."""
    print(f"Loading model: {model_name}")
    
    
    # Configure quantization if needed
    bnb_config = BitsAndBytesConfig(load_in_4bit=True) if use_quantizer else None
    
    if low_memory_load:
        # Load the model
        print("Loading model with low memory usage... e.g. inference")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
        )
    else:	
        print("Loading model with full precision...")
        model = AutoModelForCausalLM.from_pretrained(model_name)

    # Move model to appropriate device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_default_device(device)
    model = model.to(device=device)
    
    # Initialize tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Tokenizer loaded successfully")

    print(f"Model loaded and moved to {device}")
    return model, tokenizer



def save_steering_vector(steering_vector, model_name, layer_name, exp_name="", folder="./steering_vectors"):
    """
    Saves the steering vector to a file.

    Parameters:
        steering_vector (SteeringVector): The steering vector to save.
        model_name (str): The name of the model.
        layer_name (str): The name or index of the layer.
        folder (str): The folder where the steering vector will be saved. Defaults to './steering_vectors'.
    """
    # folder is relative to the main package directory, which is two levels up
    current_dir = os.path.dirname(os.path.abspath(__file__))    
    main_dir = os.path.dirname(current_dir)
    folder = os.path.join(main_dir, folder)

    # Create the folder if it doesn't exist
    os.makedirs(folder, exist_ok=True)

    if len(exp_name) > 0:
        exp_name = f"_{exp_name}"

    filename = f"{model_name.replace('/', '_')}_layer-{layer_name}{exp_name}.pt"
    filepath = os.path.join(folder, filename)
    torch.save(steering_vector, filepath)
    print(f"Steering vector saved to {filepath}")


def load_steering_vector(model_name, layer_name, exp_name="", folder="./steering_vectors"):
    """
    Loads a steering vector from a file.

    Parameters:
        model_name (str): The name of the model.
        layer_name (str): The name or index of the layer.
        folder (str): The folder where the steering vector is stored. Defaults to './steering_vectors'.

    Returns:
        SteeringVector: The loaded steering vector.
    """
    # folder is relative to the main package directory, which is two levels up
    current_dir = os.path.dirname(os.path.abspath(__file__))    
    main_dir = os.path.dirname(current_dir)
    print(f"Current directory: {main_dir}")
    folder = os.path.join(main_dir, folder)

    if len(exp_name) > 0:
        exp_name = f"_{exp_name}"
        
    filename = f"{model_name.replace('/', '_')}_layer-{layer_name}{exp_name}.pt"
    filepath = os.path.join(folder, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Steering vector file not found: {filepath}")
    steering_vector = torch.load(filepath, weights_only=False)
    print(f"Steering vector loaded from {filepath}")
    return steering_vector

