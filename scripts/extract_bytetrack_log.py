import os, sys, tqdm
import numpy as np

if __name__ == "__main__":
    with open("test_bytetrack.log", "r") as file:
        lines = file.readlines()
        result = []
        for i in range(len(lines)):
            if lines[i].startswith("OVERALL") and len(lines[i].split()) == 20:
                mota = lines[i].split()[14][:-1]
                mota = float(mota) / 100.0
                mota = round(mota, 3)
                result.append(mota)

        print(result)
        for i in range(len(result)):
            if (i + 1) %4 == 0:
                print(result[i], end="\n")
            else:
                print(result[i], end=", ")