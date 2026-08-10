# Steering Vector Framework

This repository contains a framework for optimizing and evaluating steering vectors for language models to aid LLM-judges in detecting manipulative responses.

## Dataset and Judge results:
The manipulation dataset is stored in :

The results for the GPT4.1-nano on the sycophancy and manipulation dataset, and Claude3.5-haiku on the manipulation dataset are stored in the folder `results/final_judge_results/`. For each dataset and judge there is a separate file containing the target LLM responses and the scores provided by the LLM judge. 
To rerun the visualizations from the paper, use the notebook: `viz_judge_results_paper.ipynb` under `./notebooks`

Human annotation results are stored in `results/human_annotation/`, divided over split A and B, each containing the results for 65 questions. The responses are annonymized per split so that the prolific ids are substituted by annotator_1, or other numbers, so that it is still retrievable which responses belonged to the same annotator.
To load and process the human annotator results, check out `notebooks/load_human_annoation_results.ipynb`

### Inter-annotator agreement
Krippendorff's alpha (plus Fleiss' kappa and raw percent agreement), combined and per manipulation category:

```bash
python -m steering_vec_functions.human_annotation.annotator_agreement
```

This prints the tables and writes `results/human_annotation/inter_annotator_agreement.json`. Use `--n_boot 0` to skip the bootstrap confidence intervals (the run takes ~50s with them, ~1s without), and `--leave_out_cats ...` to restrict the scope to a subset of categories. The metrics themselves live in `steering_vec_functions/human_annotation/agreement_metrics.py` and are numpy-only; running that file directly checks them against the published worked example. Section 3 of `notebooks/load_human_annoation_results.ipynb` calls the same functions.

Two companion scripts cover the rest of the human evaluation statistics:

```bash
# correctness means and Wilcoxon signed-rank tests between base / provoked / steered-provoked
python -m steering_vec_functions.human_annotation.response_quality_stats

# marginal rates, the Figure 3 headline percentages, and human vs LLM-judge cross-method tests
python -m steering_vec_functions.human_annotation.agreement_diagnostics
```

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
git clone [repo link]
cd judge_with_steered_response

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
- Different datasets (`manipulation`, `sycophancy`)
- Model configurations
- Steering vector parameters
- Output format options

The evaluation script provides multiple evaluation scenarios that can be selected individually or run together.

## Output

Results are saved as JSON files containing both the generated responses and their evaluation metrics, including manipulation/sycophancy scores and correctness assessments.
