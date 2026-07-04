import re
import matplotlib.pyplot as plt
import os
import math

# Define regex to match required fields (starting after "]" )
pattern = re.compile(r"""
    \]\s(?P<type>\w+\sEpoch)\s*:\s*(?P<epoch>\d+)\s*
    Loss:\s*(?P<loss>\d+\.\d+)\s*
    (?:lr:(?P<lr>\d+\.\d+)\s*)?
    (?:PSNR:(?P<psnr>-?\d+\.\d+)\s*)?
    (?:mePSNR:(?P<mepnsr>\d+\.\d+)\s*)?
    (?:Bpp:(?P<bpp>\d+\.\d+)\s*)?
    (?:mvBpp:(?P<mvbpp>\d+\.\d+)\s*)?
    (?:time:(?P<time>\d+\.\d+)\s*)?
    (?:index:(?P<index>\d+)\s*)?
""", re.VERBOSE)
# pattern_train = re.compile(r"""

def parse_file(file_path):
    p = pattern
    results = []
    with open(file_path, 'r') as file:
        for line in file:
            match = p.search(line)
            if match:
                results.append(match.groupdict())
    return results

def draw_loss_pic(x, y, input_data_list, file_name, base_path, filters=None, types=None, isdB=True):
    plt.figure(figsize=(20, 10))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    if types:
        marker_lst = [ 'x' if x == 'cross' else '' for x in types]
        linestyle_lst = [ '' if x == 'cross' else '-' for x in types]
    else:
        lenth = len(input_data_list)
        marker_lst = ['' for _ in range(lenth)]
        linestyle_lst = ['-' for _ in range(lenth)]
    
    for i, input_data in enumerate(input_data_list):
        if filters:
            filtered_data = [entry for entry in input_data if all(entry.get(filter_name) == filter_value for filter_name, filter_value in filters.items())]
        else:
            filtered_data = input_data

        data_x = [int(entry[x]) for entry in filtered_data if entry['loss']]
        if isdB:
            data_y = [float(entry[y]) for entry in filtered_data if entry['loss']]
        else:
            data_y = [10 * math.log10(float(entry[y])) for entry in filtered_data if entry['loss']]
        
        plt.plot(data_x, data_y, marker=marker_lst[i], linestyle=linestyle_lst[i], color=colors[i % len(colors)], label=base_path[i])
    
    plt.xlabel(x)
    
    if isdB:
        plt.ylabel(y + '(dB)')
        plt.title(f'{x} vs. {y}(dB)')
    else:
        plt.ylabel(y)
        plt.title(f'{x} vs. {y}')
    plt.grid(True)
    plt.legend()
    plt.savefig(file_name)
    plt.close()

if __name__ == '__main__':
    root_path = '/path/to/output/figures'
    file_paths = [
        "/path/to/training/log.txt",
        "/path/to/training/log2.txt"
    ]
    # types = ["line", "line", "cross", "cross", "cross", "cross"]
    types = ["line", "line"]
    pattern_type = "valid"

    assert len(file_paths) == len(types)
    base_path = [os.path.basename(x) for x in file_paths]

    parsed_data_list = [parse_file(file_path) for file_path in file_paths]

    for i in range(4):
        path = os.path.join(root_path, f'loss_fig_{str(i)}.png')
        filters = {'index': str(i)} if pattern_type == "valid" else {'type': 'Train Epoch'}
        draw_loss_pic('epoch', 'loss', parsed_data_list, path, base_path=base_path, filters=filters, types=types)
