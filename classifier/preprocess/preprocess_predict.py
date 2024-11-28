import os
import json
import jsonlines
from preprocess_utils import *

# Get the dataset name from environment variable
dataset_name = os.getenv('DATASET', 'hotpotqa')  # Default to 'hotpotqa' if not set

# Only load the dataset that you need (hotpotqa in this case)
if dataset_name == 'hotpotqa':
    orig_hotpotqa_file = os.path.join("processed_data", dataset_name, 'test_subsampled.jsonl')
    lst_hotpotqa = prepare_predict_file(orig_hotpotqa_file, dataset_name)

    # Combine data (only hotpotqa data here)
    lst_total_data = lst_hotpotqa

    # Save the results
    output_path = os.path.join("classifier", "data", f'{dataset_name}_predictions')
    save_json(output_path+'/predict.json', lst_total_data)

else:
    print(f"Dataset {dataset_name} is not supported or not available.")
