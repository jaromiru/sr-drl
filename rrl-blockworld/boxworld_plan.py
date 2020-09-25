import gym, random
import numpy as np
import copy

from astar import AStar
from tqdm import tqdm

def _find_stack(stacks, item):
	for stack_id, stack in enumerate(stacks):
		if stack[0] == item:
			return stack, stack_id

	return None, None

def _get_state_string(state):
	return [tuple(o) for o in state]

def _state_eq(s1, s2):
	s1 = {tuple(o) for o in s1}
	s2 = {tuple(o) for o in s2}

	return s1 == s2

def _to_set(s):
	return frozenset(tuple(o) for o in s)

def _to_list(s):
	return [np.array(o) for o in s]
 
def _move(s, what, where):
	if what == where:
		return None

	stack_from, stack_from_id = _find_stack(s, what)
	if stack_from is None:	 # invalid action
		return None

	s_ = copy.copy(s)

	if where == 0: 			 # to the ground, create a new stack
		stack_to = np.empty(0, dtype=np.int)
		s_.append(stack_to)
		stack_to_id = len(s_) - 1
	else: 					 
		stack_to, stack_to_id = _find_stack(s, where)

	if stack_to is None:	 # invalid action
		return None

	# move the item
	s_[stack_from_id] = np.delete(stack_from, 0)
	s_[stack_to_id]   = np.insert(stack_to, 0, what)

	# delete a potentially empty stack
	if len(s_[stack_from_id]) == 0:
		del s_[stack_from_id]

	return _to_set(s_)

class BoxworldPlan(AStar):
	def heuristic_cost_estimate(self, s, g):
		h = 0

		def common_chars(a, b):
			indx = 1
			while indx <= min(len(a), len(b)):
				if a[-indx] != b[-indx]:
					return indx - 1

				indx += 1

			return indx - 1

		for sx in s:
			found = False
			# print(f"{sx=}", end=": ")

			for gx in g:
				# find the final stack
				if sx[-1] == gx[-1]:
					# count the correct items
					found = True
					cmn = common_chars(sx, gx)
					h += len(sx) - cmn
					# print(h)
					break

			if not found:
				h += len(sx)
				# print(f"! {h}")

		return h

	def distance_between(self, n1, n2):
		return 1

	def neighbors(self, node):
		# tqdm_main.update()

		ngbrs = []
		s = _to_list(node)

		for x1 in s:
			n1 = x1[0]
			for x2 in s + [[0]]:
				n2 = x2[0]
				s_ = _move(s, n1, n2)

				if s_ is not None:
					ngbrs.append(s_)

		return ngbrs

	def is_goal_reached(self, current, goal):
		return current == goal

	def plan(self, start, goal):
		s = _to_set(start)
		g = _to_set(goal)

		return self.astar(s, g)

if __name__ == '__main__':
	from boxworld import BoxworldEnv

	env = BoxworldEnv(box_num_obj=11, box_max_steps=100)
	env.reset()

	planner = BoxworldPlan()

	state = frozenset({(1, 3), (2,), (4,)})
	goal = frozenset({(1, 2, 4, 3)})

	# for s in s_: 
	# 	h = planner.heuristic_cost_estimate(s, g)
	# 	print(h)
	# exit()

	state = env.state
	goal = env.goal

	print(state, goal)

	# tqdm_main = tqdm()
	path, path_len = planner.plan(state, goal)

	print("---")
	print(list(path))
	print(path_len)
