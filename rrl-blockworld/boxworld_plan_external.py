from tqdm import tqdm
import sys, glob, re, os, subprocess, tempfile

# requires pddl-astar from https://gitlab.com/danfis/cpddl 
# 
# git clone https://gitlab.com/danfis/cpddl.git
# cd cpddl
# ./scripts/build.sh 
# cp bin/pddl-astar ../rrl-boxworld/_plan/


class BoxworldPlanExt():
	def _to_list(self, stack):
		stack_str = []

		stack_str.append(f"(free n{stack[0]})")
		for i in range(len(stack) - 1):
			stack_str.append(f"(box-on n{stack[i]} n{stack[i+1]})")

		stack_str.append(f"(box-on n{stack[-1]} GROUND)")

		return " ".join(stack_str)


	def _get_pddl(self, objects, init, goal):
		return f'''(define (problem boxworld-pddl)
	(:domain boxworld)
	(:objects {objects})
	(:init {init} )
	(:goal (and {goal} )
	)
)
'''		

	def _run_planner(self, fname):
		cmd = f"_plan/pddl-astar -H lmcut _plan/boxworld_domain.pddl {fname}"
		# cmd = f'_plan/fast-downward.py _plan/boxworld_domain.pddl {fname} --search "astar(ipdb())"'
		p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		outs, errs = p.communicate()

		out = outs.decode()
		# print(out, errs.decode())

		# pddl-astar
		path_len = re.search(r"^;; Length: (\d+)", out, re.MULTILINE).group(1)

		# fast-downward
		# path_len = re.search(r"Plan length: (\d+)", out, re.MULTILINE).group(1)
		path = re.findall(r"^\(move .*\)", out, re.MULTILINE)

		path_len = int(path_len)

		return path, path_len
	
	def plan(self, start, goal):
		all_nodes = " ".join(["n"+str(item) for sublist in start for item in sublist])
		start_stacks = " ".join([self._to_list(x) for x in start])
		goal_stacks = " ".join([self._to_list(x) for x in goal])

		pddl = self._get_pddl(all_nodes, start_stacks, goal_stacks)

		with tempfile.NamedTemporaryFile("w+t") as file:
			file.write(pddl)
			file.flush()

			# print(start, "->", goal)
			# print(pddl)

			path, path_len = self._run_planner(file.name)

		return path, path_len

if __name__ == '__main__':
	from boxworld import BoxworldEnv

	env = BoxworldEnv(box_num_obj=9, box_max_steps=100)
	env.reset()

	planner = BoxworldPlanExt()

	# state = frozenset({(1, 3), (2,), (4,)})
	# goal = frozenset({(1, 2, 4, 3)})

	# for s in s_: 
	# 	h = planner.heuristic_cost_estimate(s, g)
	# 	print(h)
	# exit()

	state = env.state
	goal = env.goal

	# tqdm_main = tqdm()
	path, path_len = planner.plan(state, goal)

	print("---")
	print(state, "->", goal)

	print( path_len, path )

	# print(f"Env plan len {env.path_len}")
	# env_path = "\n".join([str(set(x)) for x in env.path])
	# print(env_path)
