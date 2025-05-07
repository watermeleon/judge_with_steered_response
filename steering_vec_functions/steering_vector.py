import os
import torch
import steering_opt
from typing import List, Dict, Any, Tuple

from steering_vec_functions.datasets import format_question
from steering_vec_functions.model_utils import save_steering_vector, load_steering_vector


class SteeringVector:
    """Class to handle steering vector optimization and application."""
    
    def __init__(self, model, tokenizer, layer=15, generation_length = 20, temperature=0.0):
        self.model = model
        self.tokenizer = tokenizer
        self.layer = layer
        self.vector = None
        self.loss_info = None
        self.generation_length = generation_length 
        self.temperature = temperature
    
    def optimize(self, prompt, incorrect_completion, correct_completion, 
                max_iters=20, lr=0.1, debug=False, max_norm=None) -> Tuple[torch.Tensor, Dict]:
        """Optimize a steering vector using a fixed example."""
        incorrect_completion = [incorrect_completion] if incorrect_completion is not None else []
        correct_completion = [correct_completion] if correct_completion is not None else []

        datapoint = steering_opt.TrainingDatapoint(
            format_question(prompt, self.tokenizer),
            src_completions=incorrect_completion,
            dst_completions=correct_completion,
        )
        
        self.vector, self.loss_info = steering_opt.optimize_vector(
            self.model,
            [datapoint],
            self.layer,
            tokenizer=self.tokenizer,
            max_iters=max_iters,
            lr=lr,
            debug=debug,
            max_norm=max_norm,
        )
        return self.vector, self.loss_info
    
    def optimize_from_samples(self, samples, max_iters=20, lr=0.1, max_norm=None, debug=False):
        """Optimize a steering vector using multiple samples."""
        datapoints = []
        for prompt, src_completions, dst_completions in samples:
            datapoint = steering_opt.TrainingDatapoint(
                format_question(prompt, self.tokenizer),
                src_completions=[src_completions],
                dst_completions=[dst_completions],
            )
            datapoints.append(datapoint)
            
        self.vector, self.loss_info = steering_opt.optimize_vector(
            self.model,
            datapoints,
            self.layer,
            tokenizer=self.tokenizer,
            max_iters=max_iters,
            lr=lr,
            debug=debug,
            max_norm= max_norm,
        )
        return self.vector, self.loss_info
    
    def get_response(self, question, max_tokens=None):
        """Get a normal (unsteered) response from the model."""
        if max_tokens is None:
            max_tokens = self.generation_length

        formatted_question = format_question(question, self.tokenizer)
        input_ids = self.tokenizer(formatted_question, return_tensors='pt').input_ids
        
        generated_tokens = self.model.generate(input_ids, max_new_tokens=max_tokens, do_sample=True, temperature=self.temperature)
        generated_tokens_only = generated_tokens[:, input_ids.shape[-1]:]
        
        return self.tokenizer.batch_decode(generated_tokens_only, skip_special_tokens=True)[0]
    
    def get_steered_response(self, question, max_tokens=None):
        """Get a steered response using the optimized vector."""
        if self.vector is None:
            raise ValueError("Steering vector not optimized yet. Call optimize() first.")
        if max_tokens is None:
            max_tokens = self.generation_length

        formatted_question = format_question(question, self.tokenizer)
        steering_hook = (self.layer, steering_opt.make_steering_hook_hf(self.vector))
        
        with steering_opt.hf_hooks_contextmanager(self.model, [steering_hook]): 
            input_ids = self.tokenizer(formatted_question, return_tensors='pt').input_ids
            generated_tokens = self.model.generate(input_ids, max_new_tokens=max_tokens, do_sample=True, temperature=self.temperature)
        
        generated_tokens_only = generated_tokens[:, input_ids.shape[-1]:]
        return self.tokenizer.batch_decode(generated_tokens_only, skip_special_tokens=True)[0]
    
    def save(self, folder="./results/steering_vectors", model_name=None, layer_name=None):
        """Save the steering vector to a file."""
        
        if self.vector is None:
            raise ValueError("No steering vector to save. Call optimize() first.")
        # def save_steering_vector(steering_vector, model_name, layer_name, exp_name="", folder="./steering_vectors"):

        # return save_steering_vector(self.vector, filepath, model_name, layer_name or str(self.layer))
        return save_steering_vector(self.vector, model_name, layer_name or str(self.layer), folder=folder)
    
    def load(self, folder="./results/steering_vectors", model_name=None, layer_name=None):
        """Load a steering vector from a file."""
        # def load_steering_vector(model_name, layer_name, exp_name="", folder="./steering_vectors"):

        self.vector = load_steering_vector(model_name, layer_name or str(self.layer), folder=folder)
        return self.vector
