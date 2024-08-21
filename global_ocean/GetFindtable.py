"""""
A lookup table is created to correct the results of model ensemble further, 
correcting for the underestimation of SWH by the model for more than 10, 
the corrections are all based on the training set
"""""


import numpy as np
import os
from tqdm import tqdm

def compute_bias_for_file(npz_file):
    print(npz_file)
    # Initialize arrays to store the sum of biases and counts for intervals from 0 to 16 with a step of 0.1
    bias_sums = np.zeros(161)  
    counts = np.zeros(161)  

    # Load the NPZ file containing model outputs and ground truth labels
    data = np.load(npz_file)
    out_data = data['out_data'].reshape(-1)
    label_data = data['label_data'].reshape(-1)

    # Filter valid data within the range (0, 16] for both model output and label
    valid_mask = (out_data > 0) & (out_data <= 16)
    out_data = out_data[valid_mask]
    label_data = label_data[valid_mask]

    # Calculate the bias between the label and the model output
    biases = label_data - out_data
    
    # Convert model outputs to interval indices (0 to 160) based on their value
    indices = np.floor(out_data * 10).astype(int)  
    
    # Free memory by deleting large arrays that are no longer needed
    del out_data, label_data
    
    # Accumulate biases and counts for each interval
    for i in tqdm(range(161)):
        mask = indices == i
        if np.any(mask):
            bias_sums[i] += np.sum(biases[mask])
            counts[i] += np.sum(mask)

    return bias_sums, counts


def main(data_dir):
    # List all NPZ files in the specified directory
    npz_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npz')])

    # Initialize arrays to accumulate overall bias sums and counts
    total_bias_sums = np.zeros(161)
    total_counts = np.zeros(161)

    # Process each NPZ file to compute and accumulate biases and counts
    for npz_file in npz_files:
        biases, counts = compute_bias_for_file(npz_file)
        total_bias_sums += biases
        total_counts += counts

    # Calculate the final average biases for each interval, handling cases with zero counts
    final_biases = np.divide(total_bias_sums, total_counts, where=total_counts != 0)

    return final_biases


if __name__ == '__main__':
    data_dir = r'./results/'
    lookup_table = main(data_dir)
    print("Lookup table created.")
    np.savez('lookup_table.npz', lookup_table=lookup_table)
    print(lookup_table)
