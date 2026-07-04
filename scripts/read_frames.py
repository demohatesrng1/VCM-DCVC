import json
import matplotlib.pyplot as plt
import os

import numpy as np

def standardize_data(data):
    """
    Standardize array data so that each column has mean 0 and std 1.
    
    Args:
        data (list of lists): Original data, rows are samples, columns are features
    
    Returns:
        standardized_data: Standardized data
    """
    # Convert to NumPy array for processing
    data_np = np.array(data)
    
    # Calculate mean and std for each column
    means = np.mean(data_np, axis=0)
    stds = np.std(data_np, axis=0)
    
    # Standardize data
    standardized_data = (data_np - means) / stds
    
    return standardized_data.tolist()

def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def prepare_data(data, fields):
    prepared_data = {field: [item[field] for item in data] for field in fields}
    return prepared_data

def normalize(data, fields):
    out = [standardize_data(x) for x in data]
    return out

import matplotlib.pyplot as plt
import os

def plot_data(data, fields, save_path):
    data_0 = range(len(data[0]))

    plt.figure(figsize=(10, 6))
    y_data = data

    for i, y in enumerate(y_data):
        label = fields[i]  # fields[0] is the x-axis label, fields[i+1] is the y-axis label
        plt.plot(data_0, y, label=label, marker='')

    # Add legend, title, and labels
    plt.legend()
    plt.title('data_vis')
    plt.xlabel(fields[0])
    plt.ylabel('relative value')
    plt.grid()

    # Save figure
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    plt.savefig(save_path)
    plt.close()


def main():
    # 1. Read JSON file
    file_path = '/path/to/output.json'
    out_root = "/path/to/visualization/output"
        
    # 2. Define fields to use
    fields = ['frame_bpp', 'frame_psnr']

    if not os.path.exists(out_root):
        os.makedirs(out_root)

    data = read_json(file_path)
    for key1 in data.keys():            # dataset
        for key2 in data[key1].keys():  # video
            for key3, data3 in data[key1][key2].items():    # index
                # 3. Extract required fields
                prepared_data = [data3[x] for x in fields]
                
                # 4. Normalize data
                normalized_data = normalize(prepared_data, fields)
                
                # 5. Plot and save figure
                save_path = f'{key1}_{key2}_{key3}.png'
                save_path = os.path.join(out_root, save_path)
                plot_data(normalized_data, fields, save_path)

if __name__ == "__main__":
    main()
