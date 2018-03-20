"""
input: list of float
"""
import numpy as np

def normalize_minmax(data_in, new_min = 0, new_max = 100):
    """
    input:
        data_in: list of float
        new_min: int
        new_max: int
    output:
        out: list of float
    """
    data_in = np.array(data_in)
    old_min = min(data_in)
    old_max = max(data_in)
    print(old_min, old_max)
    out = (data_in - old_min) * (new_max - new_min)/ (old_max - old_min) + new_min
    return list(out)
