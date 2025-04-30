#!/usr/bin/env python3
"""
Fine-tune OpenAI model on Hinglish dataset for code-switching Voice AI
"""

import os
import openai
import argparse
import json
from datetime import datetime

# Set up command line arguments
parser = argparse.ArgumentParser(description='Fine-tune OpenAI model on Hinglish dataset')
parser.add_argument('--dataset', type=str, default='dataset.jsonl', help='Path to the JSONL dataset file')
parser.add_argument('--model', type=str, default='gpt-3.5-turbo', help='Base model to fine-tune')
parser.add_argument('--n_epochs', type=int, default=3, help='Number of training epochs')
parser.add_argument('--learning_rate_multiplier', type=float, default=None, help='Learning rate multiplier')
parser.add_argument('--suffix', type=str, default=None, help='Suffix for fine-tuned model name')
args = parser.parse_args()

# Set API key
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")

def validate_jsonl(file_path):
    """Validate JSONL dataset format"""
    print(f"Validating dataset: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        line_number = 0
        for line in f:
            line_number += 1
            try:
                entry = json.loads(line)
                if not isinstance(entry, dict):
                    raise ValueError(f"Line {line_number}: Expected JSON object")
                if 'prompt' not in entry:
                    raise ValueError(f"Line {line_number}: Missing 'prompt' field")
                if 'completion' not in entry:
                    raise ValueError(f"Line {line_number}: Missing 'completion' field")
            except json.JSONDecodeError:
                raise ValueError(f"Line {line_number}: Invalid JSON")
    
    print(f"Dataset validation successful: {line_number} examples found")
    return line_number

def main():
    # Validate dataset
    num_examples = validate_jsonl(args.dataset)
    
    # Generate model suffix if not provided
    model_suffix = args.suffix
    if not model_suffix:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_suffix = f"hinglish_{timestamp}"
    
    # Upload dataset file
    print(f"Uploading dataset file: {args.dataset}")
    with open(args.dataset, "rb") as file:
        response = openai.File.create(
            file=file,
            purpose="fine-tune"
        )
    file_id = response.id
    print(f"File uploaded with ID: {file_id}")
    
    # Prepare fine-tuning parameters
    fine_tune_params = {
        "training_file": file_id,
        "model": args.model,
        "n_epochs": args.n_epochs,
        "suffix": model_suffix
    }
    
    # Add optional learning rate if specified
    if args.learning_rate_multiplier:
        fine_tune_params["learning_rate_multiplier"] = args.learning_rate_multiplier
    
    # Create fine-tuning job
    print("Creating fine-tuning job with parameters:")
    for k, v in fine_tune_params.items():
        print(f"  {k}: {v}")
    
    try:
        response = openai.FineTuningJob.create(**fine_tune_params)
        print("\nFine-tuning job created successfully!")
        print(f"Job ID: {response.id}")
        print(f"Status: {response.status}")
        print(f"Model: {response.model}")
        print("\nYou can check the status of your fine-tuning job with:")
        print(f"  openai api fine_tunes.get -i {response.id}")
        print("\nOr by running:")
        print(f"  openai api fine_tunes.follow -i {response.id}")
        
        # Save fine-tuning job details to file for reference
        with open(f'fine_tune_job_{response.id}.json', 'w') as f:
            json.dump({
                "job_id": response.id,
                "model": response.model,
                "status": response.status,
                "created_at": response.created_at,
                "dataset": args.dataset,
                "examples": num_examples,
                "n_epochs": args.n_epochs,
                "suffix": model_suffix,
            }, f, indent=2)
        
    except Exception as e:
        print(f"Error creating fine-tuning job: {e}")

if __name__ == "__main__":
    main()