import numpy as np

def shapes(dict):
    for p in dict:
        print(f'{p} shape {dict[p].shape}')