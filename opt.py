
import pyomo.environ as pyo
import numpy as np
import itertools

def opt_model(GridMap, empty_cells):
	''' 
	Function to create the pyomo concrete model with all 
	its parameters, variables, constraints and dummy objective

	Args: 
	-	GridMap: object of class GridMap


	Returns:
	-	m: pypmo optimisation model

	'''

	print(' * Generating optimisation model parameters')

	# Initialise model
	m = pyo.ConcreteModel()

	# Create parameters width and height and mode (row or col)
	m.C = list(range(0,GridMap._w))
	m.R = list(range(0,GridMap._h))
	m.M = ['R', 'C']

	# Create parameter row/column rules (list of lists)
	m.col_rules = GridMap._col_rules
	m.row_rules = GridMap._row_rules

	# Create parameter number of rules per row/col
	m.no_col_groups = [len(l) for l in GridMap._col_rules]
	m.no_row_groups = [len(l) for l in GridMap._row_rules]

	# Create parameter total coloured cells per row/col
	m.no_col_cells = [sum(l) for l in GridMap._col_rules]
	m.no_row_cells = [sum(l) for l in GridMap._row_rules]

	m.Ir = list(range(0, max(m.no_row_groups)+1))
	m.Ic = list(range(0, max(m.no_col_groups)+1))

	# Create variables that determine if cells are coloured
	m.x = pyo.Var(m.R, m.C, within=pyo.Binary)

	# Create variables that determine if a cell is the first coloured cell of a group
	m.y = pyo.Var(m.R, m.C, m.M, within=pyo.Binary)

	# Create variables that determine if a cell is the last coloured cell of a group
	m.z = pyo.Var(m.R, m.C, m.M, within=pyo.Binary)

	# Create auxiliary variables lambda
	m.lr = pyo.Var(m.R, m.C, m.Ir, within=pyo.Binary)
	m.lc = pyo.Var(m.R, m.C, m.Ic, within=pyo.Binary)

	# Deviation between target black cells and solution
	m.devr = pyo.Var(m.R)
	m.devc = pyo.Var(m.C)

	print(' * Generating optimisation model constraints')

	# Group of constraints to define first cell of a group (for rows and columns)
	m.first_group_r_ctr = pyo.ConstraintList()
	for r in m.R:
	    for c in m.C:
	        if c > 0:
	            m.first_group_r_ctr.add(m.y[r,c,'R'] >= m.x[r,c] - m.x[r,c-1])
	            m.first_group_r_ctr.add(m.y[r,c,'R'] <= 2 - m.x[r,c] - m.x[r,c-1])
	            m.first_group_r_ctr.add(m.y[r,c,'R'] <= m.x[r,c])
	        else:
	            m.first_group_r_ctr.add(m.y[r,c,'R'] == m.x[r,c])

	m.first_group_c_ctr = pyo.ConstraintList()
	for c in m.C:
	    for r in m.R:
	        if r > 0:
	            m.first_group_c_ctr.add(m.y[r,c,'C'] >= m.x[r,c] - m.x[r-1,c])
	            m.first_group_c_ctr.add(m.y[r,c,'C'] <= 2 - m.x[r,c] - m.x[r-1,c])
	            m.first_group_c_ctr.add(m.y[r,c,'C'] <= m.x[r,c])
	        else:
	            m.first_group_c_ctr.add(m.y[r,c,'C'] == m.x[r,c])
	            

	# Group of constraints to define last cell of a group (for rows and columns)
	m.last_group_r_ctr = pyo.ConstraintList()
	for r in m.R:
	    for c in m.C:
	        if c < m.C[-1]:
	            m.last_group_r_ctr.add(m.z[r,c,'R'] >= m.x[r,c] - m.x[r,c+1])
	            m.last_group_r_ctr.add(m.z[r,c,'R'] <= 2 - m.x[r,c] - m.x[r,c+1])
	            m.last_group_r_ctr.add(m.z[r,c,'R'] <= m.x[r,c])
	        else:
	            m.last_group_r_ctr.add(m.z[r,c,'R'] == m.x[r,c])

	m.last_group_c_ctr = pyo.ConstraintList()
	for c in m.C:
	    for r in m.R:
	        if r < m.R[-1]:
	            m.last_group_c_ctr.add(m.z[r,c,'C'] >= m.x[r,c] - m.x[r+1,c])
	            m.last_group_c_ctr.add(m.z[r,c,'C'] <= 2 - m.x[r,c] - m.x[r+1,c])
	            m.last_group_c_ctr.add(m.z[r,c,'C'] <= m.x[r,c])
	        else:
	            m.last_group_c_ctr.add(m.z[r,c,'C'] == m.x[r,c])
	            
	            
	# Rule to define the number of groups of coloured cells in each row/col
	m.no_groups_r_ctr = pyo.ConstraintList()
	for r in m.R:
	    m.no_groups_r_ctr.add(sum(m.y[r,c,'R'] for c in m.C) == m.no_row_groups[r])

	m.no_groups_c_ctr = pyo.ConstraintList()
	for c in m.C:
	    m.no_groups_c_ctr.add(sum(m.y[r,c,'C'] for r in m.R) == m.no_col_groups[c])


	# Define lambda variables
	m.lambda_ctr = pyo.ConstraintList()
	for r in m.R:
		for c in m.C:
			# Add constraint to relate to y
			m.lambda_ctr.add(
				sum( m.y[r,q,'R'] for q in range(0,c+1) ) == 
				sum(i * m.lr[r,c,i] for i in m.Ir)
				)
			m.lambda_ctr.add(
				sum( m.y[s,c,'C'] for s in range(0,r+1) ) == 
				sum(i * m.lc[r,c,i] for i in m.Ic)
				)

			# Add constraint to ensure sum of lambda variables equals 1
			m.lambda_ctr.add( sum(m.lr[r,c,i] for i in m.Ir) == 1 )
			m.lambda_ctr.add( sum(m.lc[r,c,i] for i in m.Ic) == 1 )

			for i in range(0, max(m.no_row_groups)+1):
				if i > m.no_row_groups[r]:
					# Eliminate variables for indices greater than no of groups
					m.lambda_ctr.add(m.lr[r,c,i] == 0)
			for i in range(0, max(m.no_col_groups)+1):
				if i > m.no_col_groups[c]:
					# Eliminate variables for indices greater than no of groups
					m.lambda_ctr.add(m.lc[r,c,i] == 0)

	# Constraint on the length of groups of cells in each row/column
	m.group_len_r_ctr = pyo.ConstraintList()
	for r in m.R:
		for g in range(1,m.no_row_groups[r] + 1):
			for c in m.C:
				# Before we exceed the last cell
				if (c + m.row_rules[r][g-1] - 1) <= max(m.C):
					m.group_len_r_ctr.add( 
						m.z[r,c + m.row_rules[r][g-1] - 1,'R'] >= 
						m.y[r,c,'R'] + m.lr[r,c,g] - 1
						)
	m.group_len_c_ctr = pyo.ConstraintList()
	for c in m.C:
		for g in range(1,m.no_col_groups[c] + 1):
			for r in m.R:
				# Before we exceed the last cell
				if (r + m.col_rules[c][g-1] - 1) <= max(m.R):
					m.group_len_c_ctr.add( 
						m.z[r + m.col_rules[c][g-1] - 1,c,'C'] >= 
						m.y[r,c,'C'] + m.lc[r,c,g] - 1
						)

	# Create hard rules for partial solution
	m.partial_sol = pyo.ConstraintList()
	for r in m.R:
		for c in m.C:
			if GridMap._grid[r,c] == 1:
				m.partial_sol.add(m.x[r,c] == 1)
			if empty_cells[r,c] == 1:
				m.partial_sol.add(m.x[r,c] == 0)


	m.dev_def = pyo.ConstraintList()
	for r in m.R:
		m.dev_def.add(m.devr[r] >= sum(m.x[r,c] for c in m.C) - m.no_row_cells[r])
		m.dev_def.add(m.devr[r] >= m.no_row_cells[r] - sum(m.x[r,c] for c in m.C))

	for c in m.C:
		m.dev_def.add(m.devc[c] >= sum(m.x[r,c] for r in m.R) - m.no_row_cells[c])
		m.dev_def.add(m.devc[c] >= m.no_row_cells[c] - sum(m.x[r,c] for r in m.R))


	# Create dummy objective
	def dummy_obj(m):
	    return sum(m.devr[r] for r in m.R) + sum(m.devc[c] for c in m.C)
	m.obj = pyo.Objective(rule=dummy_obj)

	return m


def solve_model(
	model, 
	solver='cbc', 
	gap=None, 
	run_time=None, 
	tee=True):

	''' Solve optimisation problem defined in model

	Args:
	-	solver: solver used, currently only cbc supported 
	-	gap: mip gap (absolute)
	-	run_time: maximum time allowed before stopping the solver

	Returns:
	-	model: same model given to the function, now containing solutions
	-	results: info on the status of the solver, e.g. termination status
	-	optimality: boolean, true if solution is optimal
	'''

	solver = pyo.SolverFactory('cbc')

	if gap is not None:
		solver.options['allowableGap'] = gap

	if run_time is not None:
		solver.options['seconds'] = run_time

	results = solver.solve(model, tee=tee)
	optimality = results.solver.termination_condition == pyo.TerminationCondition.optimal

	return model, results, optimality


def rule_based_simplify(GridMap):
	''' Function to fix some variables using some basic
	rules, similar to how humans would approach a nonogram

	Args:
	- GridMap: object of class GridMap

	Returns:
	- Hard rules for optimizer
	'''
	empty_cells = np.zeros_like(GridMap._grid)

	# Generate overlaps in rows
	GridMap._grid, empty_cells = overlap(GridMap._grid, GridMap._row_rules, 
		GridMap._w, empty_cells)

	# Generate overlaps in columns
	GridMap._grid, empty_cells = overlap(GridMap._grid.T, GridMap._col_rules, 
		GridMap._h, empty_cells.T)

	GridMap._grid = GridMap._grid.T
	empty_cells = empty_cells.T

	# Find empty cells around finished groups (rows)
	empty_cells = blanks_around_finished_groups(GridMap._grid, 
		GridMap._row_rules, GridMap._w, empty_cells)

	empty_cells = blanks_around_finished_groups(GridMap._grid.T, 
		GridMap._col_rules, GridMap._h, empty_cells.T)
	empty_cells = empty_cells.T

	# print("partial_sol", GridMap._grid)
	print("empty cells\n", empty_cells)

	return GridMap, empty_cells


def overlap(grid, rules, dim, empty_cells):
	''' Generate overlaps between placing a group all the way to the left
	and all the way to the right

	Args:
	- grid: array where solution is being stored
	- rules: list of all rules (rows or cols)
	- dim: if row number of cols and viceversa

	Returns:
	- gridmap with hard rules
	'''

	for r, rule in enumerate(rules):

		for l_i in range(len(rule)):
			
			r_i = len(rule) - l_i - 1

			left = generate_extreme_pos(l_i, rule, dim)
			right = generate_extreme_pos(r_i, rule[::-1], dim)[::-1]
			overlap = left + right

			# Write cells with overlap to solution (black cells)
			grid[r,:] = np.where(overlap == 2, 1, grid[r,:])

			# When the overlap has the same length as the group, write
			# white cells around the group to solution
			if len(overlap[overlap == 2]) == len(left[left == 1]):
				last_idx = np.max(np.where(left == 1)[0])
				first_idx = np.min(np.where(left == 1)[0])

				if last_idx < dim -1:
					empty_cells[r,last_idx + 1] = 1
				if first_idx > 0:
					empty_cells[r,first_idx - 1] = 1

	return grid, empty_cells



def blanks_around_finished_groups(grid, rules, dim, empty_cells):

	for r, rule in enumerate(rules):

		bin_groups = []
		for k, g in itertools.groupby(np.array(grid[r,:]), lambda x: x > 0):
			bin_groups.append(np.array(list(g)))

		one_groups = [len(b) for b in bin_groups if len(b) == np.sum(b)]

		for g in one_groups:
			if np.all(g >= np.array(rule)):
				print("rule", rule)
				print("one_group", one_groups)
				print("happened. row:", r, "length:",g)
				idx_list = find_group_by_length(grid[r,:], g)

				for idx in idx_list:
					if idx > 0:
						empty_cells[r,idx - 1] = 1

					if idx + g < dim:
						empty_cells[r,idx + g] = 1
						
	return empty_cells



def generate_extreme_pos(i, rule, dim):
	''' Generate position of a group if pushed all the way to the left/right

	Args:
	- i: position of the group in rules of this row/col
	- rule: number of consecutive black cells according to this group
	- dim: if row number of cols and viceversa

	Returns:
	- ext: array with ones where this group would be in this extreme
	'''

	ext_zeros = np.repeat(0, sum([rule[j] for j in range(i)]) + i)
	ext_ones = np.repeat(1, rule[i])
	ext_fill = np.repeat(0, dim - len(ext_ones) - len(ext_zeros))
	ext = np.concatenate((ext_zeros, ext_ones, ext_fill), axis=None)

	return ext


def find_group_by_length(y, target_len):
    # Setup "sentients" on either sides to make sure we have setup
    # "ramps" to catch the start and stop for the edge islands
    # (left-most and right-most islands) respectively
    y_ext = np.r_[False,y==1, False]

    # Get indices of shifts, which represent the start and stop indices
    idx = np.flatnonzero(y_ext[:-1] != y_ext[1:])

    # Lengths of islands if needed
    lens = idx[1::2] - idx[:-1:2]

	# Return all the indices that have the desired length
    return [a for a,b in zip(idx[:-1:2], lens) if b == target_len]


