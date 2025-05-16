import json
import os
from typing import Dict, List, Tuple, Optional, Union, Any




class AIManipulationDataset:
    """
    A class to manage a dataset for AI manipulation detection.
    
    This dataset contains base questions and manipulative instructions to generate
    manipulative questions, as well as functionality to store and retrieve responses.
    """
    
    def __init__(self, data_source: Union[str, Dict[str, Any]], 
                 manipulation_level: str = "subtle", short_version: bool = False, instruct_short_response=True):
        """
        Initialize the dataset from either a JSON file or a dictionary.
        
        Args:
            data_source: Either a path to a JSON file or a dictionary containing the dataset
            manipulation_level: Which level of manipulation to use ("subtle" or "strong")
        """
        if isinstance(data_source, str) and os.path.exists(data_source):
            with open(data_source, 'r') as f:
                self.data = json.load(f)
        elif isinstance(data_source, dict):
            self.data = data_source
        else:
            raise ValueError("data_source must be either a valid file path or a dictionary")
            
        self.set_manipulation_level(manipulation_level)
        self.short_version = short_version
        
        # Storage for responses
        self.responses = {}

        if instruct_short_response:
            self.prompt_suffix_base = "INSTRUCTION: you are a helpful assistant, answer the question (max 10 sentences):\n"
            self.prompt_suffix_manipulative = "Answer the question (max 10 sentences):\n"
        else:
            # use empty string
            self.prompt_suffix_base = ""
            self.prompt_suffix_manipulative = ""
        
    def set_manipulation_level(self, level: str):
        """
        Set the manipulation level to either "subtle" or "strong".
        
        Args:
            level: The manipulation level ("subtle" or "strong")
        """
        if level not in ["subtle", "strong"]:
            raise ValueError("Manipulation level must be either 'subtle' or 'strong'")
        self.manipulation_level = level
        
    def get_base_questions(self) -> List[Dict[str, str]]:
        """
        Get a list of all base questions across all categories.
        
        Returns:
            A list of dictionaries containing question text and metadata
        """
        base_questions = []

        for category in self.data["manipulation_categories"]:

            cat_questions = category["example_questions"]
            if self.short_version:
                cat_questions = cat_questions[:3]

            
            for question in cat_questions:
                prompt_q = f"{self.prompt_suffix_base} {question}"
                base_questions.append(prompt_q)
                
        return base_questions
    
    def get_manipulative_questions(self) -> List[Dict[str, str]]:
        """
        Get a list of all questions paired with manipulative instructions.
        
        Returns:
            A list of dictionaries containing question text, instruction, and metadata
        """
        manipulative_questions = []
        full_data = []

        
        for category in self.data["manipulation_categories"]:
            category_id = category["category_id"]
            higher_level_category = category["higher_level_category"]
            instruction = category["manipulative_instruction"][self.manipulation_level]
            expected_behavior = category["expected_behavior"]
            
            
            cat_questions = category["example_questions"]
            if self.short_version:
                cat_questions = cat_questions[:3]

            for question in cat_questions:
                prompt_q = f"INSTRUCTION:{instruction} {self.prompt_suffix_manipulative}\n {question}"
                manipulative_questions.append(prompt_q)

                full_data.append({
                    "question_text": question,
                    "instruction": instruction,
                    "category_id": category_id,
                    "higher_level_category": higher_level_category,
                    "expected_behavior": expected_behavior,
                    "manipulation_level": self.manipulation_level
                })
                
        return manipulative_questions, full_data
    
    def store_response(self, question_text: str, 
                       is_manipulative: bool, 
                       response: str,
                       metadata: Optional[Dict[str, Any]] = None):
        """
        Store a response for a given question.
        
        Args:
            question_text: The text of the question
            is_manipulative: Whether this is a response to a manipulative question
            response: The text of the response
            metadata: Optional metadata about the response
        """
        if question_text not in self.responses:
            self.responses[question_text] = {
                "base": None,
                "manipulative": None
            }
            
        response_type = "manipulative" if is_manipulative else "base"
        
        self.responses[question_text][response_type] = {
            "text": response,
            "metadata": metadata or {}
        }
        
        # If this is a manipulative response, add the manipulation level
        if is_manipulative:
            self.responses[question_text][response_type]["metadata"]["manipulation_level"] = self.manipulation_level
    
    def get_response(self, question_text: str, is_manipulative: bool = False) -> Optional[Dict[str, Any]]:
        """
        Get a stored response for a given question.
        
        Args:
            question_text: The text of the question
            is_manipulative: Whether to get the manipulative or base response
            
        Returns:
            A dictionary containing the response text and metadata, or None if not found
        """
        if question_text not in self.responses:
            return None
            
        response_type = "manipulative" if is_manipulative else "base"
        return self.responses[question_text].get(response_type)
    
    def get_all_questions_with_responses(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all questions that have at least one response.
        
        Returns:
            A dictionary mapping question text to response data
        """
        return self.responses
    
    def get_evaluation_pairs(self) -> List[Dict[str, Any]]:
        """
        Get pairs of base and manipulative responses for the same questions.
        Only includes questions that have both types of responses.
        
        Returns:
            A list of dictionaries containing question text and both responses
        """
        evaluation_pairs = []
        
        for question, responses in self.responses.items():
            if responses["base"] and responses["manipulative"]:
                # Find category metadata for this question
                category_info = {}
                for q in self.get_base_questions():
                    if q["question_text"] == question:
                        category_info = {
                            "category_id": q["category_id"],
                            "higher_level_category": q["higher_level_category"]
                        }
                        break
                
                evaluation_pairs.append({
                    "question": question,
                    "base_response": responses["base"]["text"],
                    "manipulative_response": responses["manipulative"]["text"],
                    "manipulation_level": responses["manipulative"]["metadata"].get("manipulation_level", self.manipulation_level),
                    **category_info
                })
                
        return evaluation_pairs
    
    def save_to_json(self, output_path: str):
        """
        Save the dataset, including all responses, to a JSON file.
        
        Args:
            output_path: Path to save the JSON file
        """
        output_data = {
            "manipulation_categories": self.data["manipulation_categories"],
            "responses": self.responses,
            "current_manipulation_level": self.manipulation_level
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
    
    @classmethod
    def load_with_responses(cls, input_path: str) -> 'AIManipulationDataset':
        """
        Load a dataset including responses from a JSON file.
        
        Args:
            input_path: Path to the JSON file
            
        Returns:
            A new AIManipulationDataset object with loaded responses
        """
        with open(input_path, 'r') as f:
            data = json.load(f)
            
        # Extract the manipulation categories and create a new dataset
        dataset = cls({"manipulation_categories": data["manipulation_categories"]}, 
                     data.get("current_manipulation_level", "subtle"))
        
        # Load the responses
        dataset.responses = data.get("responses", {})
        
        return dataset
