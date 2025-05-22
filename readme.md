# Steering Vector Framework

This repository contains a framework for optimizing and evaluating steering vectors for language models to aid LLM-judges in detecting manipulative responses.

## Dataset and Judge results:
The manipulation dataset is stored in :

The results for the GPT4.1-nano on the sycophancy and manipulation dataset, and Claude3.5-haiku on manipulation dataset are stored in the folder `results/final_judge_results/`:
```
judged_responses_manipulation_Claude_FINAL.json
judged_responses_manipulation_GPT4Nano_FINAL.json
judged_responses_sycophancy_GPT4Nano.json
```
To rerun the vizualizations from the paper use the notebook: `viz_judge_results_paper.ipynb` under `./notebooks`


# Minimal Setup

This project provides tools for generating and evaluating AI model responses using steering vectors, with specific focus on detecting manipulation and sycophancy behaviors.

## Prerequisites

- Python 3.8 or higher
- PyTorch
- Transformers
- OpenAI API key (for response evaluation)

## Installation

```bash
# Clone the repository
git clone [repository-url]
cd [repository-name]

# Install required dependencies
pip install -r requirements.txt
```

## Basic Usage

The workflow consists of two main steps: generating responses with steering vectors and evaluating those responses.

### Step 1: Generate Responses

Generate steered responses using the following command:

```bash
python -u -m steering_vec_functions.generate_provoked_and_steered_responses \
    --data_set manipulation \
    --num_samples 20 \
    --num_iters 20 \
    --lr 0.1 \
    --low_memory_load \
    --generation_length 100 \
    --use_load_vector
```

This will create a JSON file containing the generated responses in the `results/responses/` directory.

### Step 2: Evaluate Responses

Evaluate the generated responses using:

```bash
python -m steering_vec_functions.judge.judge_responses_three_judges \
    --input_file results/responses/your_generated_response_file.json \
    --openai_model gpt-4.1-nano \
    --data_type manipulation
```

This will analyze the responses and save evaluation results in the `results/judge_results/` directory.

## Configuration Options

The response generation script supports various parameters including:
- Different datasets (`manipulation`, `feedback`)
- Model configurations
- Steering vector parameters
- Output format options

The evaluation script provides multiple evaluation scenarios that can be selected individually or run together.

## Output

Results are saved as JSON files containing both the generated responses and their evaluation metrics, including manipulation/sycophancy scores and correctness assessments.