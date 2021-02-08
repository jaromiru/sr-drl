from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

def find_path(map, start, end, alg=AStarFinder):
	grid = Grid(matrix=map)
	# g_start = grid.node(start[1], start[0]) # x, y
	# g_end = grid.node(end[1], end[0])

	g_start = grid.node(*start) # x, y
	g_end = grid.node(*end)

	finder = alg()
	path, runs = finder.find_path(g_start, g_end, grid)

	return path

tdict = {
	(0, -1): 1,
	(0, 1): 2,
	(-1, 0): 3,
	(1, 0): 4
}

def tdiff(a, b):
	return (a[0] - b[0], a[1] - b[1])

def translate_path(path):
	dpath = [tdiff(path[i + 1], path[i]) for i in range(0, len(path) - 1)]
	tpath = [tdict[x] for x in dpath]

	return tpath
