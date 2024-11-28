import os
import json
from preprocess_utils import *

# Create directories if they do not exist
def ensure_dir_exists(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

# Load JSONL file in chunks, ensuring lines don't exceed a character limit
def load_json_in_chunks(file_path, char_limit=6000):
    data = []
    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            if len(line) > char_limit:
                print(f"Skipping line {i + 1} due to character limit.")
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding line {i + 1}: {e}")
    return data

# Save JSONL file
def save_jsonl(output_file, data):
    ensure_dir_exists(output_file)  # Ensure the directory exists
    with open(output_file, 'w') as file:
        for entry in data:
            file.write(json.dumps(entry) + '\n')

# Process and save HotpotQA data
def process_hotpotqa_data(input_file, output_file):
    print(f"Loading data from {input_file}...")
    json_data = load_json_in_chunks(input_file)  # Load data without limits
    print(f"Loaded {len(json_data)} valid objects from the file.")
    print(f"Saving processed data to {output_file}...")
    save_jsonl(output_file, json_data)
    return json_data

def main():
    # Paths to the input and output files
    train_input_file = os.path.join("processed_data", 'hotpotqa', 'train.jsonl')  # Adjust if needed
    train_output_file = os.path.join('classifier', "data", "hotpotqa", 'binary', 'hotpotqa_train.jsonl')

    # Step 1: Process the HotpotQA data
    json_data = process_hotpotqa_data(train_input_file, train_output_file)

    # Step 2: Save a subset (e.g., 400 samples) for testing
    subset_file = os.path.join("classifier", "data", 'hotpotqa', 'binary', 'hotpotqa_subset.jsonl')
    subset_data = json_data[:400]
    print(f"Saving a subset of {len(subset_data)} objects to {subset_file}...")
    save_jsonl(subset_file, subset_data)

    print("Processing complete.")

if __name__ == "__main__":
    main()
