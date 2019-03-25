
import pyomo.environ as pyo
import numpy as np
import itertools
from utils import initialise_dict_ranges, islandinfo
import pyutilib.subprocess.GlobalData



def opt_model(GridMap):
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
			if GridMap._grid[r,c] == 0.5:
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
	h,
	w,
	solver='cbc', 
	gap=None, 
	run_time=120, 
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
	print("*  Solving model")

	pyutilib.subprocess.GlobalData.DEFINE_SIGNAL_HANDLERS_DEFAULT = False

	solver = pyo.SolverFactory(solver)
	solver.options['seconds'] = run_time
	results = solver.solve(model, tee=tee)
	optimality = results.solver.termination_condition == pyo.TerminationCondition.optimal

	if optimality:
		print('Termination condition: optimal')

		# Create array with solution
		grid = np.zeros((h, w))
		for c in model.C:
			for r in model.R:
				grid[r,c] = model.x[r,c].value 
		return grid

	else:
		return np.zeros((h,w))


def rule_based_simplify(GridMap):
	''' Function to fix some variables using some basic
	rules, similar to how humans would approach a nonogram

	Args:
	- GridMap: object of class GridMap

	Returns:
	- Hard rules for optimizer
	'''
	range_dict = initialise_dict_ranges(GridMap)

	# Generate overlaps in rows
	range_dict = initial_ext_ind(GridMap, range_dict, "R")
	range_dict = initial_ext_ind(GridMap, range_dict, "C")

	for i in range(3):

		# Find empty cells around finished groups (rows)

		GridMap._grid = blanks_around_finished(GridMap, range_dict, "R")
		GridMap._grid = blanks_around_finished(GridMap, range_dict, "C")

		GridMap._grid = blanks_around_longest_possible(GridMap, range_dict, "R")
		GridMap._grid = blanks_around_longest_possible(GridMap, range_dict, "C")

		GridMap._grid  = update_range(GridMap, range_dict, "R")

		GridMap._grid = overlap(GridMap, range_dict)

		GridMap._grid = blanks_after_before_extreme_groups(GridMap, range_dict, "R")
		GridMap._grid = blanks_after_before_extreme_groups(GridMap, range_dict, "C")



	return GridMap



def overlap(GridMap, range_dict):
	for r in range(GridMap._h):
		for i, g in enumerate(GridMap._row_rules[r]):
			start = range_dict['R'][r]['S'][i]
			end = range_dict['R'][r]['E'][i]

			GridMap._grid[r, end - g + 1 : start + g] = 1

	for c in range(GridMap._w):
		for i, g in enumerate(GridMap._col_rules[c]):
			start = range_dict['C'][c]['S'][i]
			end = range_dict['C'][c]['E'][i]

			GridMap._grid[end - g + 1 : start + g, c] = 1

	return GridMap._grid


def initial_ext_ind(GridMap, range_dict, rc):
	''' Generate overlaps between placing a group all the way to the left
	and all the way to the right

	Args:
	- GridMap: object of class gridmap
	- rules: list of all rules (rows or cols)
	- dim: if row number of cols and viceversa

	Returns:
	- gridmap with hard rules
	'''

	if rc == "R":
		rules = GridMap._row_rules
		dim = GridMap._w
	elif rc == "C":
		rules = GridMap._col_rules
		dim = GridMap._h

	for i, rule in enumerate(rules):
		for l_i in range(len(rule)):
			
			r_i = len(rule) - l_i - 1

			range_dict[rc][i]['S'][l_i] = generate_extreme_pos(l_i, rule, dim)
			range_dict[rc][i]['E'][l_i] = dim - 1 - generate_extreme_pos(r_i, rule[::-1], dim)

	return range_dict


def blanks_after_before_extreme_groups(GridMap, range_dict, rc):
	if rc == "R":
		rules = GridMap._row_rules
		dim = GridMap._w
		grid = GridMap._grid

	elif rc == "C":
		rules = GridMap._col_rules
		dim = GridMap._h
		grid = GridMap._grid.T

	for r_i, rule in enumerate(rules):

		if rc == "C" and r_i == 6:
			print(r_i, rule)
			for g_i, g in enumerate(rule):

				left = range_dict[rc][r_i]['S'][g_i] 
				right = range_dict[rc][r_i]['E'][g_i] 

				if g_i == 1:				
					print(" *  ",g, g_i, "left", left, "right", right)
					print("       ", "len", len(rule))


					if (right - left + 1 == g) and (g_i == 0) and left > 1:
						print("CASE 1")
						grid[r_i, 0 : left - 1] = 0.5

					if (right - left + 1 == g) and (g_i == len(rule) - 1) and right < dim - 2:
						print("CASE 2")
						grid[r_i, right + 2 : -1] = 0.5

	if rc == "C":
		grid = grid.T

	return grid


def update_range(GridMap, range_dict, rc):
	'''TODO: COLUMNS'''
	# print(GridMap._grid)
	if rc == "R":
		rules = GridMap._row_rules
		dim = GridMap._w
		grid = GridMap._grid
	elif rc == "C":
		rules = GridMap._col_rules
		dim = GridMap._h 
		grid = GridMap._grid.T

	for r_i, rule in enumerate(rules):
		# print()
		# print("ROW:  ", r_i, rule)
		for g_i, g in enumerate(rule):
			start = range_dict[rc][r_i]['S'][g_i]
			end = range_dict[rc][r_i]['E'][g_i]
			empty_idx = np.where(grid[r_i,:] == 0.5)[0]


			empty_idx = empty_idx[(empty_idx >= start) & (empty_idx <= start + g - 1)]
			# print(empty_idx)
			if len(empty_idx) > 0:
				last_possible = np.max(empty_idx)
				# print("    * OLD: ", range_dict["R"][r_i]["S"][g_i], range_dict["R"][r_i]["E"][g_i])				
				range_dict["R"][r_i]["S"][g_i] = max(last_possible + 1, range_dict["R"][r_i]["S"][g_i])
				# print("    * NEW: ", range_dict["R"][r_i]["S"][g_i] + 1, range_dict["R"][r_i]["E"][g_i])		
			

	return grid


def blanks_around_longest_possible(GridMap, range_dict, rc):
	''' TBC
	'''
	if rc == "R":
		rules = GridMap._row_rules
		dim = GridMap._w
	elif rc == "C":
		rules = GridMap._col_rules
		dim = GridMap._h

	for r_i, rule in enumerate(rules):

		if rc == "R":
			rc_sol = GridMap._grid[r_i,:]
		else:
			rc_sol = GridMap._grid[:,r_i]
		
		ones_ranges, ones_lens = islandinfo(rc_sol)
		for p_i, p in enumerate(ones_ranges):
			if np.all(ones_lens[p_i] >= np.array(rule)):
				if p[0] > 0:
					if rc == "R": GridMap._grid[r_i, p[0] - 1] = 0.5
					elif rc == "C": GridMap._grid[p[0] - 1, r_i] = 0.5
				if p[1] < dim - 1: 
					if rc == "R": GridMap._grid[r_i, p[1] + 1] = 0.5
					elif rc == "C": GridMap._grid[p[1] + 1, r_i] = 0.5

	return GridMap._grid



def blanks_around_finished(GridMap, range_dict, rc):
	''' Function to find finished groups and placing empty 
	cells around them

	Args:
	- grid: array where solution is being stored

	'''

	if rc == "R":
		rules = GridMap._row_rules
		dim = GridMap._w
	elif rc == "C":
		rules = GridMap._col_rules
		dim = GridMap._h

	for r_i, rule in enumerate(rules):
		for g_i, g in enumerate(rule):

			left = range_dict[rc][r_i]['S'][g_i] 
			right = range_dict[rc][r_i]['E'][g_i] 

			if (right - left + 1 == g):
				if left > 0: 
					if rc == "R": GridMap._grid[r_i, left - 1] = 0.5
					elif rc == "C": GridMap._grid[left - 1, r_i] = 0.5
				if right < dim - 1: 
					if rc == "R": GridMap._grid[r_i, right + 1] = 0.5
					elif rc == "C": GridMap._grid[right + 1, r_i] = 0.5

	return GridMap._grid



def generate_extreme_pos(i, rule, dim):
	''' Generate position of a group if pushed all the way to the left/right

	Args:
	- i: position of the group in rules of this row/col
	- rule: number of consecutive black cells according to this group
	- dim: if row number of cols and viceversa

	Returns:
	- ext: array with ones where this group would be in this extreme
	'''

	ext_idx = sum([rule[j] for j in range(i)]) + i

	return ext_idx


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


