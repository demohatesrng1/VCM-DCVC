import matplotlib.pyplot as plt
from bjontegaard_metric import BD_RATE, BD_PSNR

Class_B = {}
Class_C = {}
Class_D = {}
Class_E = {}
UVG = {}
data_dicts = [Class_B, Class_C, Class_D, Class_E, UVG]


def read_hevc(filename, name="new"):
    # Read file content
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Initialize lists
    hevc_b = []
    hevc_c = []
    hevc_d = []
    hevc_e = []
    uvg = []
    current_list = None

    # Parse file content
    for line in lines:
        line = line.strip()
        if line.startswith("HEVC_B"):
            current_list = hevc_b
        elif line.startswith("HEVC_C"):
            current_list = hevc_c
        elif line.startswith("HEVC_D"):
            current_list = hevc_d
        elif line.startswith("HEVC_E"):
            current_list = hevc_e
        elif line.startswith("UVG"):
            current_list = uvg
        else:
            values = line.split()
            if current_list is not None and len(values) == 2:
                current_list.append((float(values[0]), float(values[1])))
    
    
    for x, y in zip([hevc_b, hevc_c, hevc_d, hevc_e, uvg], [Class_B, Class_C, Class_D, Class_E, UVG]):
        y[name] = x
        
    return 



def print_bdbr():
    for k in data_dicts[0].keys():
        if k in ban_key:
            continue
        bd_rates = []
        for data in data_dicts: 
            if "hem" in k or "HEM" in k:
                anchor_name = "HEM-official-GOP32"
            else:
                print("key: ", k)
                raise Exception("No anchor found")
            bpp = [item[0] for item in data[k]]
            psnr = [item[1] for item in data[k]]
            anchor_bpp = [item[0] for item in data[anchor_name]]
            anchor_psnr = [item[1] for item in data[anchor_name]]
            
            bd_rates.append(BD_RATE(anchor_bpp, anchor_psnr, bpp, psnr))
        print(f"Name:{k}, \tAnchor:{anchor_name}, \tBD-rate:{bd_rates[0]:.2f}%/{bd_rates[1]:.2f}%/{bd_rates[2]:.2f}%/{bd_rates[3]:.2f}%")
    return


if __name__ == '__main__':
    file_names_official = [
        ("../../results/checkpoint_output_official_gop32.txt", "HEM-official-GOP32"),
    ]

    file_names_hem = [
        ("../../results/quickresult_hem2_iter1889901_gop32.txt", "hem2_iter1889901"),
        ("../../results/quickresult_hem2_iter1986819_gop32.txt", "hem2_iter1986819"),
        ("../../results/quickresult_hem2_iter2019125_gop32.txt", "hem2_iter2019125"),
        ("../../results/quickresult_hem4_iter1801203_gop32.txt", "hem4_iter1801203"),
        ("../../results/quickresult_hem4_iter1906204_gop32.txt", "hem4_iter1906204"),
        ("../../results/quickresult_hem4_iter2003128_gop32.txt", "hem4_iter2003128"),
    ]

    file_names = file_names_official + file_names_hem
    ban_key = []

    for i, j in file_names:
        read_hevc(i, name=j)

    datasets = ['Class B', 'Class C', 'Class D', 'Class E']

    plt.grid()
    fig, ax = plt.subplots(2, 2, figsize=(20.5, 13)) # Figure size
    plt.subplots_adjust( wspace=0.18, hspace=0.30)
    plt.figure()

    for filename, data_dict, idx in zip(datasets, data_dicts, range(6)):
        i = int(idx / 2)
        j = int(idx % 2)

        marker_counter = 0
        for key in data_dict.keys():
            if key in ban_key:
                continue
            data = data_dict[key]
            bpp = [item[0] for item in data]
            psnr = [item[1] for item in data]

            marker = "o"
            linewidth = 4 if "-official" in key else 2
            linestyle = '--' if any(substring in key for substring in ["GOP12", "gop12"]) else "-"
            ax[i][j].plot(bpp, psnr, marker=marker, linestyle=linestyle, linewidth=linewidth)  # Marker size, line width
            marker_counter += 1

        # Set tick label size and font
        ax[i][j].tick_params(labelsize=22)
        labels = ax[i][j].get_xticklabels() + ax[i][j].get_yticklabels()
        [label.set_fontname('sans-serif') for label in labels]

        # Set legend and its font and size
        font1 = {
                'weight': 'normal',
                'size': 14,
                }
        ax[i][j].legend(list(data_dict.keys()), loc=4, prop=font1)

        # Set x/y axis labels and corresponding font format
        font2 = {
                'weight': 'normal',
                'size': 20,
                }
        ax[i][j].set_xlabel('Bpp', font2)
        ax[i][j].set_ylabel('PSNR (dB)', font2)

        # Set title and corresponding font format
        font3 = {
                'weight': 'normal',
                'size': 20,
                }
        ax[i][j].set_title(filename, font=font3)
        ax[i][j].grid()

    fig.savefig('./PSNR.png')
    plt.close(fig)
    print_bdbr()

