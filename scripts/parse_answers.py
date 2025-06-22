# script parses from hf dataset into csv for eval by HF Math eval 

import argparse
import csv
from datasets import load_dataset
from datasets import Dataset
import os
import re

"""
Usage: python parse_answers.py --dataset_id TheRealPilot638/Llama-3.2-1B-Instruct-vanilla-test \
                               --dataset_split train \
                               --output_file "answers.csv"
"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", type=str, required=True)
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--output_file", type=str, default="answers.csv")
    args = parser.parse_args()
    return args

def extract_boxed_answer(text):
    """
    Extract the final answer from the boxed{answer} format.
    Handles nested braces properly by counting brace pairs.
    
    Args:
        text (str): The full LLM output containing reasoning and boxed answer
        
    Returns:
        str: The extracted answer, or the original text if no boxed answer found
    """
    # Find the start of boxed{
    start_pattern = r'boxed\{'
    match = re.search(start_pattern, text)
    
    if not match:
        return text
    
    # Start position after "boxed{"
    start_pos = match.end()
    
    # Count braces to find the matching closing brace
    brace_count = 1
    current_pos = start_pos
    
    while current_pos < len(text) and brace_count > 0:
        char = text[current_pos]
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
        current_pos += 1
    
    if brace_count == 0:
        # Found matching closing brace, extract content
        return text[start_pos:current_pos-1]
    else:
        # No matching closing brace found, return original text
        return text

def to_csv(x, file_name, directory="/home/dlimpus/ece695aih_project/Math-Verify/examples", clean_answers=True):
    """
    Write dictionary data to a CSV file in the specified directory.
    Optionally clean the 'answer' field to extract only the final boxed answer.
    
    Args:
        x (dict): Dictionary with lists as values (expects 'answer' and 'gold' keys)
        file_name (str): Name of the CSV file
        directory (str): Directory path where the file should be saved (default: current directory)
        clean_answers (bool): Whether to extract boxed answers from the 'answer' field (default: True)
    """
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Combine directory and filename
    file_path = os.path.join(directory, file_name)
    
    # Create a copy of the data to avoid modifying the original
    data_to_write = x.copy()
    
    # Clean the answers if requested
    if clean_answers and 'answer' in data_to_write:
        cleaned_answers = []
        for answer in data_to_write['answer']:
            cleaned_answer = extract_boxed_answer(str(answer))
            cleaned_answers.append(cleaned_answer)
        data_to_write['answer'] = cleaned_answers
    
    with open(file_path, 'w', newline='') as csvfile:
        fieldnames = data_to_write.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        num_rows = len(data_to_write['answer'])
        
        for i in range(num_rows):
            row = {key: data_to_write[key][i] for key in fieldnames}
            writer.writerow(row)
            
        print(f"Successfully wrote {num_rows} rows to {file_path}")
    return

if __name__=="__main__":

    args = parse_args()
    data = {"answer": [], "gold": []} # gold is true answer, answer is model answer

    print(f"loading dataset: {args.dataset_id}")
    dataset = load_dataset(args.dataset_id, split=args.dataset_split)

    for sample in dataset:

        if 'answer' in sample: # this is gold
            data['gold'].append(sample['answer'])
        else:
            print(f"no solution found in example: {sample.keys()}")

        if 'Response' in sample: # this is answer
            data['answer'].append(sample['Response'])
        else:
            print(f"no answer found in example: {sample.keys()}")

    print(f"loaded {len(data['gold'])} examples from the dataset")

    to_csv(data, args.output_file)