#!/usr/bin/env python3
"""
Fine-tune OpenAI model on Hinglish dataset for code-switching Voice AI
Updated for OpenAI API v1.0+
"""

import os
import argparse
import json
from datetime import datetime
from openai import OpenAI
import time

# Set up command line arguments
parser = argparse.ArgumentParser(description='Fine-tune OpenAI model on Hinglish dataset')
parser.add_argument('--dataset', type=str, default='dataset.jsonl', help='Path to the JSONL dataset file')
parser.add_argument('--model', type=str, default='gpt-3.5-turbo', help='Base model to fine-tune')
parser.add_argument('--n_epochs', type=int, default=3, help='Number of training epochs')
parser.add_argument('--learning_rate_multiplier', type=float, default=None, help='Learning rate multiplier')
parser.add_argument('--suffix', type=str, default=None, help='Suffix for fine-tuned model name')
args = parser.parse_args()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable not set")

def convert_to_chat_format(file_path, output_path=None):
    """
    Convert legacy JSONL format to the newer chat format required by OpenAI API v1.0+
    Returns the path to the converted file and the number of examples
    """
    if not output_path:
        output_path = f"{os.path.splitext(file_path)[0]}_chat_format.jsonl"
    
    print(f"Converting dataset to chat format: {file_path} -> {output_path}")
    count = 0
    
    with open(file_path, 'r', encoding='utf-8') as f_in, open(output_path, 'w', encoding='utf-8') as f_out:
        for line_number, line in enumerate(f_in, 1):
            try:
                entry = json.loads(line)
                if not isinstance(entry, dict):
                    raise ValueError(f"Line {line_number}: Expected JSON object")
                if 'prompt' not in entry:
                    raise ValueError(f"Line {line_number}: Missing 'prompt' field")
                if 'completion' not in entry:
                    raise ValueError(f"Line {line_number}: Missing 'completion' field")
                
                # Extract user message from the prompt (removing "Assistant:" at the end)
                user_content = entry['prompt'].replace("\nAssistant:", "")
                
                # Create chat format
                chat_entry = {
                    "messages": [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": entry['completion']}
                    ]
                }
                
                # Write to output file
                f_out.write(json.dumps(chat_entry) + "\n")
                count += 1
                
            except json.JSONDecodeError:
                raise ValueError(f"Line {line_number}: Invalid JSON")
    
    print(f"Conversion successful: {count} examples processed")
    return output_path, count

def main():
    # Convert dataset to chat format required by OpenAI API v1.0+
    chat_format_file, num_examples = convert_to_chat_format(args.dataset)
    
    # Generate model suffix if not provided
    model_suffix = args.suffix
    if not model_suffix:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_suffix = f"hinglish_{timestamp}"
    
    # Upload dataset file
    print(f"Uploading dataset file: {chat_format_file}")
    with open(chat_format_file, "rb") as file:
        response = client.files.create(
            file=file,
            purpose="fine-tune"
        )
    file_id = response.id
    print(f"File uploaded with ID: {file_id}")
    
    # Wait for file processing to complete
    print("Waiting for file processing to complete...")
    while True:
        file_status = client.files.retrieve(file_id=file_id)
        if file_status.status == "processed":
            print("File processing complete!")
            break
        print("File still processing. Waiting...")
        time.sleep(5)
    
    # Prepare fine-tuning parameters
    fine_tune_params = {
        "training_file": file_id,
        "model": args.model,
        "suffix": model_suffix
    }
    
    # In OpenAI API v1.0+, n_epochs and learning_rate_multiplier are under hyperparameters
    hyperparameters = {}
    if args.n_epochs:
        hyperparameters["n_epochs"] = args.n_epochs
    
    if args.learning_rate_multiplier:
        hyperparameters["learning_rate_multiplier"] = args.learning_rate_multiplier
    
    # Only add hyperparameters if we have any
    if hyperparameters:
        fine_tune_params["hyperparameters"] = hyperparameters
    
    # Create fine-tuning job
    print("Creating fine-tuning job with parameters:")
    for k, v in fine_tune_params.items():
        print(f"  {k}: {v}")
    
    try:
        response = client.fine_tuning.jobs.create(**fine_tune_params)
        
        print("\nFine-tuning job created successfully!")
        print(f"Job ID: {response.id}")
        print(f"Status: {response.status}")
        print(f"Model: {response.model}")
        print("\nYou can check the status of your fine-tuning job using this script with the job ID")
        
        # Save fine-tuning job details to file for reference
        with open(f'fine_tune_job_{response.id}.json', 'w') as f:
            job_details = {
                "job_id": response.id,
                "model": response.model,
                "status": response.status,
                "created_at": response.created_at,
                "dataset": args.dataset,
                "chat_format_file": chat_format_file,
                "examples": num_examples,
                "n_epochs": args.n_epochs if args.n_epochs else "default",
                "suffix": model_suffix,
            }
            json.dump(job_details, f, indent=2, default=str)
        
        # Ask if user wants to monitor progress
        monitor = input("\nDo you want to monitor the fine-tuning progress? (y/n): ")
        if monitor.lower() == 'y':
            print(f"\nMonitoring fine-tuning job {response.id}...")
            while True:
                job_status = client.fine_tuning.jobs.retrieve(response.id)
                print(f"Status: {job_status.status} | {datetime.now().strftime('%H:%M:%S')}")
                
                if job_status.status in ["succeeded", "failed", "cancelled"]:
                    print(f"Job {job_status.status}!")
                    if job_status.status == "succeeded":
                        print(f"Fine-tuned model: {job_status.fine_tuned_model}")
                    break
                    
                time.sleep(60)  # Check every minute
        
    except Exception as e:
        print(f"Error creating fine-tuning job: {e}")

if __name__ == "__main__":
    main()