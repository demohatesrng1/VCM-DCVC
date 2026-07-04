import json
from collections import defaultdict
import os
import sys
import numpy as np

def load_json(file_path):
    """Load JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def calculate_averages(data):
    """Calculate average BPP and PSNR for each model in each dataset"""
    results = defaultdict(lambda: defaultdict(lambda: {'total_bpp': 0, 'total_psnr': 0, 'total_msssim': 0, 
                                                        'total_semantic_psnr': 0, 'total_semantic_msssim': 0, 
                                                        'total_frame_lpips_alexnet': 0, 
                                                        'total_semantic_entropy': 0, 'total_semantic_lpips_cnn': 0, 'total_semantic_lpips_swin': 0, 'total_semantic_lpips_dino': 0, 
                                                        'count': 0}))

    for dataset_name, files in data.items():
        for file_name, models in files.items():
            for model_num, output in models.items():
                # Ensure output is a dict and contains customized content
                if isinstance(output, dict) and 'semantic_psnrs' in output and 'semantic_msssims' in output and 'semantic_entropy' in output and 'semantic_lpips_cnn' in output and 'semantic_lpips_swin' in output and 'semantic_lpips_dino' in output:
                    results[dataset_name][model_num]['total_semantic_psnr'] += np.mean(output['semantic_psnrs'])
                    results[dataset_name][model_num]['total_semantic_msssim'] += np.mean(output['semantic_msssims'])
                    results[dataset_name][model_num]['total_semantic_entropy'] += np.mean(output['semantic_entropy'])
                    results[dataset_name][model_num]['total_semantic_lpips_cnn'] += np.mean(output['semantic_lpips_cnn'])
                    results[dataset_name][model_num]['total_semantic_lpips_swin'] += np.mean(output['semantic_lpips_swin'])
                    results[dataset_name][model_num]['total_semantic_lpips_dino'] += np.mean(output['semantic_lpips_dino'])
                    results[dataset_name][model_num]['total_frame_lpips_alexnet'] += np.mean(output['frame_lpips_alexnet'])
                    
                # Ensure output is a dict and contains 'ave_p_frame_bpp' and 'ave_p_frame_psnr'
                if isinstance(output, dict) and 'ave_all_frame_bpp' in output and 'ave_all_frame_psnr' in output:
                    results[dataset_name][model_num]['total_bpp'] += output['ave_all_frame_bpp']
                    results[dataset_name][model_num]['total_psnr'] += output['ave_all_frame_psnr']
                    results[dataset_name][model_num]['total_msssim'] += output['ave_all_frame_msssim']
                    results[dataset_name][model_num]['count'] += 1
                else:
                    print(f"Warning: Missing or incorrect format for bpp/psnr in {dataset_name} - {file_name} - {model_num}")

    averages = defaultdict(dict)
    for dataset_name, models in results.items():
        for model_num, metrics in models.items():
            if metrics['count'] > 0:
                average_bpp = metrics['total_bpp'] / metrics['count']
                average_psnr = metrics['total_psnr'] / metrics['count']
                average_msssim = metrics['total_msssim'] / metrics['count']
                average_semantic_psnr = metrics['total_semantic_psnr'] / metrics['count']
                average_semantic_msssim = metrics['total_semantic_msssim'] / metrics['count']
                average_semantic_entropy = metrics['total_semantic_entropy'] / metrics['count']
                average_semantic_lpips_cnn = metrics['total_semantic_lpips_cnn'] / metrics['count']
                average_semantic_lpips_swin = metrics['total_semantic_lpips_swin'] / metrics['count']
                average_semantic_lpips_dino = metrics['total_semantic_lpips_dino'] / metrics['count']
                average_frame_lpips_alexnet = metrics['total_frame_lpips_alexnet'] / metrics['count']
                averages[dataset_name][model_num] = {
                    'average_bpp': average_bpp,
                    'average_psnr': average_psnr,
                    'average_msssim': average_msssim,
                    'average_frame_lpips_alexnet': average_frame_lpips_alexnet, 
                    'average_semantic_psnr': average_semantic_psnr,
                    'average_semantic_msssim': average_semantic_msssim,
                    'average_semantic_entropy': average_semantic_entropy,
                    'average_semantic_lpips_cnn': average_semantic_lpips_cnn,
                    'average_semantic_lpips_swin': average_semantic_lpips_swin,
                    'average_semantic_lpips_dino': average_semantic_lpips_dino,
                }
            else:
                averages[dataset_name][model_num] = {
                    'average_bpp': None,
                    'average_psnr': None,
                    'average_msssim': None,
                    'average_frame_lpips_alexnet': None,  
                    'average_semantic_psnr': None,
                    'average_semantic_msssim': None,
                    'average_semantic_entropy': None,
                    'average_semantic_lpips_cnn': None,
                    'average_semantic_lpips_swin': None,
                    'average_semantic_lpips_dino': None,
                }
    return averages

def save_json(data, file_path):
    """Save data to JSON file"""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def save_to_txt(data, file_path):
    """Save data to TXT file"""
    with open(file_path, 'w') as f:
        for dataset_name, models in data.items():
            # Write dataset name
            f.write(f'{dataset_name}\n')                
            f.write(f'bpp, psnr, msssim, frame_lpips_alexnet, sem_psnr, sem_msssim, entropy, lpips_cnn, lpips_swin, lpips_dino\n')
            # Write bpp and psnr for each model
            for model_num, metrics in models.items():
                average_bpp = metrics.get('average_bpp', 'N/A')
                average_psnr = metrics.get('average_psnr', 'N/A')
                average_msssim = metrics.get('average_msssim', 'N/A')
                average_frame_lpips_alexnet = metrics.get('average_frame_lpips_alexnet', 'N/A')
                average_semantic_psnr = metrics.get('average_semantic_psnr', 'N/A')
                average_semantic_msssim = metrics.get('average_semantic_msssim', 'N/A')
                average_semantic_entropy = metrics.get('average_semantic_entropy', 'N/A')
                average_semantic_lpips_cnn = metrics.get('average_semantic_lpips_cnn', 'N/A')
                average_semantic_lpips_swin = metrics.get('average_semantic_lpips_swin', 'N/A')
                average_semantic_lpips_dino = metrics.get('average_semantic_lpips_dino', 'N/A')
                f.write(f'{average_bpp}, {average_psnr}, {average_msssim}, {average_frame_lpips_alexnet}, {average_semantic_psnr}, {average_semantic_msssim}, {average_semantic_entropy}, {average_semantic_lpips_cnn}, {average_semantic_lpips_swin}, {average_semantic_lpips_dino}\n')

def main(input_file_path):
    # Define input and output file paths

    output_txt_file_path = input_file_path.replace("json", "csv")

    # Load JSON file
    data = load_json(input_file_path)

    # Calculate averages
    averages = calculate_averages(data)

    # # Save results to JSON file

    # Save results to TXT file
    save_to_txt(averages, output_txt_file_path)
    print(f'The results have been saved to {output_txt_file_path}')

if __name__ == '__main__':
    input_file_path = sys.argv[1]
    main(input_file_path)
