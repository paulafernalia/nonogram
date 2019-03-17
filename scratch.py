import numpy as np

def islandinfo(y, target_len):
    # Setup "sentients" on either sides to make sure we have setup
    # "ramps" to catch the start and stop for the edge islands
    # (left-most and right-most islands) respectively
    y_ext = np.r_[False,y==1, False]

    # Get indices of shifts, which represent the start and stop indices
    idx = np.flatnonzero(y_ext[:-1] != y_ext[1:])

    # Lengths of islands if needed
    lens = idx[1::2] - idx[:-1:2]

    # Using a stepsize of 2 would get us start and stop indices for each island
    # return idx[:-1:2], lens

    return [a for a,b in zip(idx[:-1:2], lens) if b == target_len]


y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
islandinfo(y, 4)