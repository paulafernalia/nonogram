import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import colors
import plotly.graph_objs as go


class GridMap:

    ''' class that contains all information of a 
    nonogram game: size and row/col rules'''

    def __init__(self, col_rules, row_rules):
        self._w = len(col_rules)
        self._h = len(row_rules)

        self._start = [0, 0]
        self._end   = [self._w-1, 0]
        
        self._col_rules = col_rules
        self._row_rules = row_rules
        
        self._num_rules = sum([len(r) for r in row_rules]) + sum([len(c) for c in col_rules])
        self._grid = np.zeros([self._h, self._w])

        self._row_labels = [" ".join([str(r)+" " for r in rule]) for rule in row_rules]
        self._col_labels = ["\n".join([str(r) for r in rule]) for rule in col_rules]
        
    @property
    def grid(self):
        return self._grid
    
    @grid.setter
    def grid(self, value):
        self._grid = value
    
    def plot_grid(self, figsize=(8,4)):
        ''' Plot empty grid map '''

        cmap   = colors.ListedColormap(['white', 'grey', 'black'])
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
    return list(zip(idx[:-1:2], idx[1::2]-1)), lens



def plotly_heatmap(grid, w, h, row_labels, col_labels):
    data= go.Heatmap(
                # x=list(range(w)),
                # y=list(range(0,-h)),
                z=grid,
                colorscale=[
                    [0, 'rgb(255, 255, 255)'], # white
                    [0.5, 'rgb(179, 204, 255)'], # grey
                    [1, 'rgb(0, 40, 77)'] # black
                ],
                showscale=False,
                hoverinfo="none",
                xgap=2,
                ygap=2,
            )
    layout = go.Layout(
                height=30 * (h+4),
                width=30 * (w+4),
                plot_bgcolor=('#9cbff4'),
                yaxis=go.layout.YAxis(
                    ticktext=row_labels,
                    color='rgb(0, 40, 77)',
                    # family='sans-serif',
                    tickvals=np.arange(h+1),
                    ticks="",
                    zeroline=False,
                    linecolor= ('rgb(230, 238, 255)'),
                    mirror=True,
                    autorange="reversed"
                ),
                xaxis=go.layout.XAxis(
                    ticktext=col_labels,
                    tickvals=np.arange(w+1),
                    ticks="",
                    color='rgb(0, 40, 77)',
                    # family='sans-serif',
                    zeroline=False,
                    tickangle=-90,
                    linecolor= ('rgb(230, 238, 255)'),
                    mirror=True,
                )
            )

    return {'data': [data],
            'layout': layout}

def initialise_dict_ranges(GridMap):
    range_dict = {}

    range_dict['R'] = {}
    range_dict['C'] = {}

    for r in range(GridMap._h):
        range_dict['R'][r] = {}

        range_dict['R'][r]['S'] = np.repeat(0, 
                                            len(GridMap._row_rules[r]))
        range_dict['R'][r]['E'] = np.repeat(GridMap._w, 
                                            len(GridMap._row_rules[r]))

    for c in range(GridMap._w):
        range_dict['C'][c] = {}

        range_dict['C'][c]['S'] = np.repeat(0, 
                                            len(GridMap._col_rules[c]))
        range_dict['C'][c]['E'] = np.repeat(GridMap._w, 
                                            len(GridMap._col_rules[c]))

    return range_dict
