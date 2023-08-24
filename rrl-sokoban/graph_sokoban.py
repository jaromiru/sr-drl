from gym_sokoban.envs.sokoban_env import SokobanEnv
from gym_sokoban.envs.boxoban_env import BoxobanEnv
import pathfinder
import numpy as np
import random, gym

from config import config

class GraphSokoban(gym.Env):
	def __init__(self, **kwargs):
		# different processes need different seeds
		random.seed()
		np.random.seed()

		self.pos_feats = kwargs.pop('pos_feats', False)

		boxoban = kwargs.pop('boxoban', False)
		soko_size = kwargs.pop('soko_size')
		soko_boxes = kwargs.pop('soko_boxes')

		if boxoban:
			self.env = BoxobanEnv(**kwargs)

		else:
			kwargs.pop('split', None)
			kwargs.pop('difficulty', None)
			kwargs.pop('subset', None)

			self.env = SokobanEnv(**kwargs, dim_room=soko_size, num_boxes=soko_boxes)


# class GraphSokoban(SokobanEnv):
# 	def __init__(self, **kwargs):
# 		# different processes need different seeds
# 		random.seed()
# 		np.random.seed()

# 		kwargs.pop('split', None)
# 		kwargs.pop('difficulty', None)
# 		self.env.__init__(**kwargs, dim_room=config.soko_size, num_boxes=config.soko_boxes)

	def find_path(self, start, end): # in x,y coordinates
		xy_dim = self.env.dim_room[::-1]

		start = np.array(start)
		end = np.array(end)

		if np.any(start < 0) or np.any(end < 0) or np.any(start >= xy_dim) or np.any(end >= xy_dim):
			return None # the coordinates are out of the map

		arr_block = 1 - ((self.env.room_fixed == 0) + (self.env.room_state == 4) + (self.env.room_state == 3)).view(np.int8)

		path = pathfinder.find_path(arr_block, start, end)
		tpath = pathfinder.translate_path(path)

		return tpath

	ACTIONS = {
		0: "MOVE_TO",
		1: "PUSH_UP",
		2: "PUSH_DOWN",
		3: "PUSH_LEFT",
		4: "PUSH_RIGHT"
	}

	PUSH_ACTION = {
		1: np.array( (0, 1) ),
		2: np.array( (0, -1) ),
		3: np.array( (1, 0) ),
		4: np.array( (-1, 0) )
	}

	EDGE_LEFT  = [1, 0, 0, 0]
	EDGE_RIGHT = [0, 1, 0, 0]
	EDGE_UP    = [0, 0, 1, 0]
	EDGE_DOWN  = [0, 0, 0, 1]

	def _get_ppos(self):
		return self.env.player_position[::-1] # x, y now

	def _execute(self, tpath):
		if not tpath:
			return None, 0, None, None

		r_tot = 0

		for a in tpath:
			s, r, d, i = self.env.step(a)
			r_tot += r

			# env.render(mode='human')
			# input()

		return s, r_tot, d, i

	# the edge features and index are static
	# def _create_edges(self):
	# 	nnodes = self.dim_room[0] * self.dim_room[1]

	# 	# nrow = self.dim_room[0]
	# 	ncol = self.dim_room[1]

	# 	def row(x):
	# 		return x // ncol

	# 	# create edge index
	# 	edge_index_left  = [(i, i - 1) for i in range(nnodes) if row(i - 1) == row(i)]
	# 	edge_index_right = [(i, i + 1) for i in range(nnodes) if row(i + 1) == row(i)]
	# 	edge_index_up    = [(i, i - ncol) for i in range(nnodes) if (i - ncol) >= 0]
	# 	edge_index_down  = [(i, i + ncol) for i in range(nnodes) if (i + ncol) < nnodes]

	# 	edge_index = np.concatenate([edge_index_left, edge_index_right, edge_index_up, edge_index_down]).T

	# 	# create edge features
	# 	E_LEFT  = [1, 0, 0, 0]
	# 	E_RIGHT = [0, 1, 0, 0]
	# 	E_UP    = [0, 0, 1, 0]
	# 	E_DOWN  = [0, 0, 0, 1]

	# 	e_left  = [E_LEFT for i in range(nnodes) if row(i - 1) == row(i) ]
	# 	e_right = [E_RIGHT for i in range(nnodes) if row(i + 1) == row(i) ]
	# 	e_up    = [E_UP for i in range(nnodes) if (i - ncol) >= 0]
	# 	e_down  = [E_DOWN for i in range(nnodes) if (i + ncol) < nnodes]

	# 	e_feats = np.concatenate([e_left, e_right, e_up, e_down])

	# 	return e_feats, edge_index

	def _create_edges(self, node_mask, count):
		node_ids = np.full_like(node_mask, -1, dtype=np.int64)
		node_ids[node_mask] = np.arange(count)

		# use the fact that there are always wall on the borders
		# hence we don't have to check for overflow
		def get_dir(ids, dr, axis, e_feat):
			shift = np.roll(ids, dr, axis)
			edges = np.stack([ids, shift], axis=2)

			edges_mask = edges != -1
			edges_mask = edges_mask[:, :, 0] * edges_mask[:, :, 1]
			edges = edges[edges_mask]

			feats = [e_feat] * len(edges) 

			return edges, feats

		# TODO: possible 2x speedup, left is revese of right, up is reverse of down
		e_left, f_left	  = get_dir(node_ids, -1, 1, self.EDGE_LEFT)
		e_right, f_right  = get_dir(node_ids, 1, 1, self.EDGE_RIGHT)
		e_up, f_up	   	  = get_dir(node_ids, -1, 0, self.EDGE_UP)	
		e_down, f_down	  = get_dir(node_ids, 1, 0, self.EDGE_DOWN)

		edge_index = np.concatenate([e_left, e_right, e_up, e_down]).T
		edge_feats = np.concatenate([f_left, f_right, f_up, f_down])

		return edge_feats, edge_index

	def _to_graph(self, s):
		# create node features
		s_shape = s[0].shape

		if self.pos_feats:
			yl, xl = s_shape[0], s_shape[1]
			# pos = np.mgrid[-1:1:complex(s_shape[0]), -1:1:complex(s_shape[1])] # <- positional features
			pos = np.mgrid[0:yl-1:complex(yl), 0:xl-1:complex(xl)] / 10. # <- positional features
		else:
			pos = np.zeros((0, *s_shape))	# <- no positional features

		walls, goals, boxes, player = s

		n_feats = np.stack([goals, boxes, player], axis=0)
		n_feats = np.concatenate([n_feats, pos], axis=0)
		n_feats = np.moveaxis(n_feats, 0, 2)

		# remove the walls
		no_walls = walls == 0
		no_walls_indices = np.argwhere(no_walls)

		n_feats = n_feats[no_walls]

		# flatten
		if self.pos_feats:
			n_feats = np.reshape(n_feats, (-1, 5))
		else:
			n_feats = np.reshape(n_feats, (-1, 3))

		# ---- get edges
		edge_feats, edge_index = self._create_edges(no_walls, len(no_walls_indices))

		step_idx = self.env.num_env_steps / self.env.max_steps	# use the elementary step_idx!

		return n_feats, edge_feats, edge_index, step_idx, no_walls_indices

	def step(self, action):
		self.step_idx += 1

		pos, a = action
		ppos = self._get_ppos()

		a += 1 # disable the move action (mapping [0 - 4] to [1 - 4])

		el_steps_start = self.env.num_env_steps
		
		if a == 0:
			tpath = self.find_path(ppos, pos)

			if tpath:	# the path exists
				s, r_tot, d, i = self._execute(tpath)
			else:
				s, r_tot, d, i = self.env.step(0)		 # if it's not possible do noop

		else:
			pos += self.PUSH_ACTION[a]
			tpath = self.find_path(ppos, pos)

			if tpath or np.all(ppos == pos): # if there is a path or we are in the right location already
				_, r_tot, _, _ = self._execute(tpath)	 # move to position such that the block can be moved
				s, r, d, i = self.env.step(a) 			 # push the block in the right direction
				r_tot += r

			else:
				s, r_tot, d, i = self.env.step(0)		 # if it's not possible do noop

		i['elementary_steps'] = self.env.num_env_steps - el_steps_start

		d_true = self.env._check_if_all_boxes_on_target()	# the environment finishes
		d_new = d_true or self.env._check_if_maxsteps() 	# timeout should not be considered in the learning

		s_true = self._to_graph(s)

		if d_new:
			s_new = self.reset()
		else:
			s_new = s_true

		i['s_true'] = s_true
		i['d_true'] = d_true

		return s_new, r_tot, d_new, i

	def _raw_step(self, a):
		return self.env.step(a)

	def reset(self):
		s = self.env.reset()
		self.step_idx = 0

		return self._to_graph(s)

	def render(self, mode, **kwargs):
		kwargs['mode'] = mode
		return self.env.render(**kwargs)

# def random_box():
# 	env = GraphSokoban(num_boxes=3)

# 	while True:
# 		env.render(mode='human')

# 		random_box = np.argwhere(env.room_state == 4)[0][::-1] #x, y
# 		print(f"pushing box at {random_box} left")
# 		input()

# 		env.step(random_box, 3)
# 		env.render(mode='human')
# 		print("Finished")
# 		input()

# 		env.reset()


# def raw_manual():
	# from getkey import getkey, keys

	# env = GraphSokoban(dim_room=(5,3), num_boxes=1)

	# while True:
	# 	env.render(mode='human')
	# 	raw = env.render(mode='raw')

	# 	print(raw)
	# 	env._to_graph(raw)


	# 	key = getkey()
	# 	key_dict = {keys.UP: 1, keys.DOWN: 2, keys.LEFT: 3, keys.RIGHT: 4}

	# 	if key in key_dict:
	# 		a = key_dict[key]
	# 	else:
	# 		a = 0 # no op

	# 	print(a)
	# 	env._raw_step(a)
	# 	# super(SokobanEnv, env).step(a)

def save_img():
	soko_size = (10, 10)
	soko_boxes = 4

	# soko_size = (15, 15)

	filename = f"{soko_size[0]}x{soko_size[1]}-{soko_boxes}box.png"

	env = GraphSokoban(soko_size=soko_size, soko_boxes=soko_boxes, boxoban=True)
	env.reset()

	from PIL import Image
	# env.render(mode='human')
	s_img = env.render(mode='rgb_array')

	alpha = Image.fromarray(s_img, "RGB")
	alpha.save(filename)


def graph_manual():
	print("Write action: xpos ypos action, i.e. 2 2 1")
	print(GraphSokoban.ACTIONS)

	env = GraphSokoban(soko_size=(5,5), soko_boxes=3)
	env.reset()

	while True:
		env.render(mode='human')

		x, y, a = [int(x) for x in input().split()]
		# x = 1
		# y = 1
		# a = 1
		s, r, d, i = env.step(((x, y), a))

		print(s)
		print(r)
		print(d, i)

if __name__ == '__main__':
	# random_box()
	# raw_manual()
	# graph_manual()
	save_img()