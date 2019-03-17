import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import colors


class GridMap:

    ''' class that contains all information of a 
    nonogram game: size and row/col rules'''

    def __init__(self, col_rules, row_rules):
        self._w = len(col_rules)
        self._h = len(row_rules)

        self._start = [0,   0]
        self._end   = [self._w-1, 0]
        
        self._col_rules = col_rules
        self._row_rules = row_rules
        
        self._num_rules = sum([len(r) for r in row_rules]) + sum([len(c) for c in col_rules])
        self._grid = np.zeros([self._h, self._w])
        
    @property
    def grid(self):
        return self._grid
    
    @grid.setter
    def grid(self, value):
        self._grid = value
    
    def plot_grid(self, figsize=(8,4)):
        ''' Plot empty grid map '''

        cmap   = colors.ListedColormap(['white', 'black'])
        norm   = colors.BoundaryNorm([2,1,-1], cmap.N)

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(self._grid, cmap=cmap)

        # draw gridlines
        ax.grid(which='major', axis='both', linestyle='-', color='grey', linewidth=2)
        ax.set_xticks(np.arange(0.5, self._w, 1));
        ax.set_yticks(np.arange(0.5, self._h, 1));

        # hide axes labels
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        ax.set_title("Solution")
        plt.show()

        return(fig, ax)



def islandinfo(y):
    # Setup "sentients" on either sides to make sure we have setup
    # "ramps" to catch the start and stop for the edge islands
    # (left-most and right-most islands) respectively
    y_ext = np.r_[False,y==1, False]

    # Get indices of shifts, which represent the start and stop indices
    idx = np.flatnonzero(y_ext[:-1] != y_ext[1:])

    # Lengths of islands if needed
    lens = idx[1::2] - idx[:-1:2]

    # Using a stepsize of 2 would get us start and stop indices for each island
    return zip(idx[:-1:2], idx[1::2]-1), lens

