from matplotlib import colors

# Workaround to use matplotlib in virtualenv
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import random
import pyomo.environ as pyo
import numpy as np
import time
import itertools
# %matplotlib notebook

# import rules
from sample_puzzle import col_rules, row_rules

print(col_rules)