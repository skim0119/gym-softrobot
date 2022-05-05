import numpy as np

def do_normalization(data, limit):
    return (data - limit[0]) / (limit[1] - limit[0])
