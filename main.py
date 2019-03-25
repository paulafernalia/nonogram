import numpy as np

from utils import GridMap
import sample_puzzle
from opt import opt_model, solve_model, rule_based_simplify

# Create game instance
game = GridMap(col_rules, row_rules)

game, empty_cells = rule_based_simplify(game)

# Create optimisation model
model = opt_model(game, empty_cells)

# Plot solution
fig, ax = game.plot_grid()

# Solve problem
model, results, optimality = solve_model(model, tee=True)

if optimality:
	print('Termination condition: optimal')

# Create array with solution
game.grid = np.zeros((game._h,game._w))
for c in model.C:
    for r in model.R:
        game.grid[r,c] = model.x[r,c].value    

for i in range(0,3):
	lambda_sol = np.zeros((game._h,game._w))
	for c in model.C:
	    for r in model.R:
	        lambda_sol[r,c] = round(model.lr[r,c,i].value,2)   

# Plot solution
fig, ax = game.plot_grid()
