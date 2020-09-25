import gym, random, copy
import numpy as np

from config import config
from boxworld_plan import BoxworldPlan
from boxworld_plan_external import BoxworldPlanExt

def _get_random_state(num_obj):
	stacks = []
	obj = np.arange(1, num_obj + 1)

	while len(obj) > 0:
		stack_len = np.random.randint(1, len(obj)+1)
		stack = np.random.choice(obj, stack_len, replace=False)

		stacks.append(stack)
		obj = np.setdiff1d(obj, stack)

	return stacks

def _gen_unique_embeddings(emb_len, num_obj):
	while True:
		ground_object = np.zeros(emb_len)

		objects = [ground_object]
		objects.extend( [np.random.choice([-1., 0., 1.], emb_len) for x in range(num_obj)] )

		obj_set = {tuple(o) for o in objects}

		if len(obj_set) == num_obj + 1:
			return np.array(objects)

def _find_stack(stacks, item):
	for stack_id, stack in enumerate(stacks):
		if stack[0] == item:
			return stack, stack_id

	return None, None

def _get_state_string(state):
	return [tuple(o) for o in state]

class BoxworldEnv(gym.Env):
	REWARD_FINISHED = 10
	REWARD_STEP = -0.1

	def __init__(self, **kwargs):
		random.seed()
		np.random.seed()

		self.num_obj = kwargs["box_num_obj"]
		self.max_steps = kwargs["box_max_steps"]

		if 'plan' in kwargs and kwargs['plan'] is not None:
			self.plan = True

			if kwargs['plan'] == 'astar':
				self.planner = BoxworldPlan()
			elif kwargs['plan'] == 'external':
				self.planner = BoxworldPlanExt()

		else:
			self.plan = False

	def _move(self, what, where):
		# print(f"{what}->{where}")
		if what == where:
			print("!invalid action what==where")
			return

		stack_from, stack_from_id = _find_stack(self.state, what)
		if stack_from is None:	 # invalid action
			print("!invalid action cannot move what")
			return

		if where == 0: 			 # to the ground, create a new stack
			stack_to = np.empty(0, dtype=np.int)
			self.state.append(stack_to)
			stack_to_id = len(self.state) - 1
		else: 					 
			stack_to, stack_to_id = _find_stack(self.state, where)

		if stack_to is None:	 # invalid action
			print("!invalid action cannot move to where")
			return

		# move the item
		self.state[stack_from_id] = np.delete(stack_from, 0)
		self.state[stack_to_id]   = np.insert(stack_to, 0, what)

		# delete a potentially empty stack
		if len(self.state[stack_from_id]) == 0:
			del self.state[stack_from_id]

	def step(self, action):
		# print("State:", _get_state_string(self.state), " | Goal:", _get_state_string(self.goal), " | Action:", action, " | Steps:", self.steps)

		self._move(*action)

		self.steps += 1
		goal_reached = self._check_goal_reached()
		steps_exceeded = self.steps >= self.max_steps
		done = goal_reached or steps_exceeded

		info = {
			'd_true': goal_reached,
			'done': done,
			'steps': self.steps,
			'raw_start': self.start,
			'raw_state': self.state,
			'raw_goal': self.goal
		}

		if self.plan:
			info['path_len'] = self.path_len
			info['path'] = self.path

		if goal_reached:
			s_ = self.reset()
			s_true = s_ # does not matter
			reward = self.REWARD_STEP + self.REWARD_FINISHED

		else:
			if steps_exceeded:
				s_true = self._get_state()
				s_ = self.reset()
			else:
				s_ = self._get_state()
				s_true = s_
	
			reward = self.REWARD_STEP

		info['s_true'] = s_true

		return s_, reward, done, info

	def reset(self):
		self.steps = 0

		# self.objs  = _gen_unique_embeddings(self.emb_len, self.num_obj)
		self.objs = np.ones((self.num_obj + 1, 1)) # + 1 ground
		self.objs[0] = 0

		while True:
			self.start = _get_random_state(self.num_obj)
			self.goal  = _get_random_state(self.num_obj)

			self.state = copy.copy(self.start)

			if not self._check_goal_reached():	# ensure that the state and goal differ
				break

		# determine the optimal solution
		if self.plan:
			self.path, self.path_len = self.planner.plan(self.state, self.goal)

		return self._get_state()

	def _check_goal_reached(self):
		state = {tuple(o) for o in self.state}
		goal = {tuple(o) for o in self.goal}

		return state == goal

	def _get_state(self):
		state_node_feats, state_edge_feats, state_edge_index = self._to_graph(self.state, True)
		goal_node_feats, goal_edge_feats, goal_edge_index = self._to_graph(self.goal, False)

		node_feats = state_node_feats
		edge_feats = np.concatenate( (state_edge_feats, goal_edge_feats) )
		edge_index = np.concatenate( (state_edge_index, goal_edge_index), axis=1)

		free_boxes = [x[0] for x in self.state]

		return node_feats, edge_feats, edge_index, free_boxes

	def _to_graph(self, state, is_state):
		if is_state:
			edge_up = [[0,  1]]; edge_down = [[0, -1]]
		else:
			edge_up = [[1,  1]]; edge_down = [[1, -1]]

		def stack_edges(stack):
			shift = np.roll(stack, -1)

			e_down = np.stack([stack, shift])
			e_down[1, -1] = 0      # ground object

			e_up = np.flip(e_down)
			e_feats = edge_down * len(shift) + edge_up * len(shift)

			return np.concatenate([e_down, e_up], axis=1), e_feats

		edge_index = []
		edge_feats = []

		for stack in state:
			e_index, e_feats = stack_edges(stack)

			edge_index.append(e_index)
			edge_feats.append(e_feats)

		edge_index = np.concatenate(edge_index, axis=1)
		edge_feats = np.concatenate(edge_feats)
		node_feats = self.objs

		return node_feats, edge_feats, edge_index

if __name__ == '__main__':
	env = BoxworldEnv(box_num_obj=10, box_max_steps=100)
	s = env.reset()

	while True:
		print(f"state = {_get_state_string(env.state)}")
		print(f"goal  = {_get_state_string(env.goal)}")

		print("<from> <to>: ", end="")
		n_from, n_to = [int(x) for x in input().split()]

		s, r, d, i = env.step((n_from, n_to))
		# print(f"{s=} {r=}, {d=}, {i=}")
