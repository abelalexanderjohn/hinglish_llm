#!/usr/bin/env python3
"""
Test fine-tuned Hinglish model with sample prompts
Updated for OpenAI API v1.0+
"""

import os
import argparse
import json
from datetime import datetime
from openai import OpenAI

# Define test prompts for evaluating the model
DEFAULT_TEST_PROMPTS = [
    "Mujhe ek chai pilao",
    "Weather kaisa hai today in Delhi?",
    "Meeting reschedule karna hai, help karo",
]

# Set up command line arguments
parser = argparse.ArgumentParser(description='Test fine-tuned Hinglish model')
parser.add_argument('--model', type=str, required=True, help='Name of the fine-tuned model')
parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature (0.0-1.0)')
parser.add_argument('--max_tokens', type=int, default=150, help='Maximum response length')
parser.add_argument('--prompts_file', type=str, help='JSON file with test prompts')
parser.add_argument('--output_file', type=str, help='File to save results')
args = parser.parse_args()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable not set")

def load_test_prompts(file_path=None):
    """Load test prompts from file or use defaults"""
    if file_path and os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            prompts = json.load(f)
            if isinstance(prompts, list):
                return prompts
            else:
                print(f"Warning: Expected list in {file_path}, using default prompts")
    return DEFAULT_TEST_PROMPTS

def generate_response(model, user_text, temperature=0.7, max_tokens=150):
    """Generate a response using the fine-tuned model"""
    try:
        # For all modern fine-tuned models (v1.0+ API only uses chat completion format)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": user_text}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating response: {e}")
        return f"Error: {str(e)}"

def main():
    # Load test prompts
    test_prompts = load_test_prompts(args.prompts_file)
    print(f"Testing fine-tuned model: {args.model}")
    print(f"Temperature: {args.temperature}, Max tokens: {args.max_tokens}")
    print(f"Testing with {len(test_prompts)} prompts\n")
    
    results = []
    
    # Process each test prompt
    for i, prompt_text in enumerate(test_prompts, 1):
        print(f"Prompt {i}: {prompt_text}")
        
        # Generate response
        response = generate_response(
            args.model, 
            prompt_text, 
            args.temperature, 
            args.max_tokens
        )
        
        print(f"Response: {response}\n")
        
        # Store result
        results.append({
            "prompt": prompt_text,
            "response": response
        })
    
    # Save results if output file specified
    if args.output_file:
        output_data = {
            "model": args.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "timestamp": datetime.now().isoformat(),
            "results": results
        }
        
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()