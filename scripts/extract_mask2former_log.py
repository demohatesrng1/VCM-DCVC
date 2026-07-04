import os, sys, tqdm
import numpy as np

if __name__ == "__main__":
    with open("test_mask2former.log", "r") as file:
        result = []
        lines = file.readlines()
        for i in range(len(lines)):
            if "d2.evaluation.testing]: copypaste: AP,AP50,AP75,APs,APm,APl,AR1,AR10" in lines[i]:
                tmp = lines[i+1].split(": copypaste: ")[1].split(",")[0]
                tmp = float(tmp)
                result.append(tmp)
        for i in range(len(result)):
            if (i + 1) %4 == 0:
                print(result[i], end="\n")
            else:
                print(result[i], end=", ")