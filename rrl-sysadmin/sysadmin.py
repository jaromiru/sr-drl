import gym, random, copy, string, uuid
import numpy as np

rddl_template = string.Template('''
non-fluents nf_sysadmin_inst_$uid {
	domain = sysadmin_mdp;
	objects {
		computer : {$objects};
	};
	non-fluents {
		REBOOT-PROB = $reboot_prob;
		$connections
	};
}

instance sysadmin_inst_$uid {
	domain = sysadmin_mdp;
	non-fluents = nf_sysadmin_inst_$uid;
	init-state {
		$running
	};

	max-nondef-actions = $maxactions;
	horizon  = $horizon;
	discount = $discount;
}
''')

# ----------------------------------------------------------
class SysAdminEnv(gym.Env):
	REBOOT_PROB = 0.04
	REBOOT_PENALTY = 0.75	# IDEA: change?
	MAX_CONNECTIONS = 3

	def __init__(self, offset=0, save_domain=False, **kwargs):
		random.seed()
		np.random.seed()

		self.num_obj = kwargs["env_num_obj"]
		self.max_steps = kwargs["env_max_steps"]

		self.offset = offset # first-time initialize with random actions
		self.save_domain = save_domain

		self.multi = kwargs["multi"]

	def step(self, actions): 
		running_ = self.running.copy()

		# update the running nodes
		for c in range(self.num_obj):
			if self.running[c]:
				conns = self.connections[0, (self.connections[1] == c)]	 # connections to this node
				n_conns = len(conns)
				n_conns_running = np.sum(self.running[conns])

				# up_prob = 0.45 + 0.5 * (1 + n_conns_running) / (1 + n_conns)
				up_prob = 0.9 * (1 + n_conns_running) / (1 + n_conns) 	# IDEA: change?
				running_[c] = np.random.binomial(1, up_prob)

			else:
				running_[c] = np.random.binomial(1, self.REBOOT_PROB)

		# restart the selected nodes
		if len(actions) != 0:
			running_[actions] = 1

		reward = np.sum(self.running) - self.REBOOT_PENALTY * len(actions)
		self.reward_total += reward
		self.running = running_

		# compute stats
		self.steps += 1
		done = self.steps >= self.max_steps
		s_true = self._get_state()

		info = {
			'd_true': False,
			'done': done,
			'steps': self.steps,
			's_true': s_true,
			'num_obj': self.num_obj,
			'reward_total': self.reward_total
		}

		if done:
			s_ = self.reset()
		else:
			s_ = s_true

		return s_, reward, done, info

	def reset(self):
		self.steps = 0
		self.reward_total = 0.
		self.running = np.ones(self.num_obj)

		# generate random connections
		self.connections = []

		# IDEA: better graphs?
		for node_a in range(self.num_obj):
			possible_connections = np.delete( np.arange(self.num_obj), node_a )
			conns_ids = np.random.choice(possible_connections, np.random.randint(1, self.MAX_CONNECTIONS), replace=False)

			conns = np.stack([ np.full(len(conns_ids), node_a), conns_ids ])

			self.connections.append(conns)
			# self.connections.append(np.flip(conns, axis=0))

		self.connections = np.concatenate(self.connections, axis=1)
		self.connections = np.unique(self.connections, axis=1)

		# first-time init
		if self.offset > 0:
			offset = self.offset % self.max_steps
			self.offset = 0

			for i in range(offset):
				self.step([]) # noop

		if self.save_domain:
			uid = uuid.uuid4().hex
			fn = f"_plan/sysadmin_inst_{uid}.rddl"
			rddl = self._get_rddl(uid)

			with open(fn, 'wt') as f:
				f.write(rddl)

		return self._get_state()

	def _get_state(self):
		node_feats = self.running.reshape(-1, 1)
		edge_feats = None

		return node_feats, edge_feats, self.connections


	def _get_rddl(self, uid):
		objects = ",".join([f"c{i}" for i in range(self.num_obj)])
		connections = " ".join([f"CONNECTED(c{x[0]},c{x[1]});" for x in self.connections.T])
		running = " ".join([f"running(c{i});" for i, x in enumerate(self.running)])
		max_actions = self.num_obj if self.multi else 1

		rddl = rddl_template.substitute(uid=uid, objects=objects, maxactions=max_actions, reboot_prob=self.REBOOT_PROB, connections=connections, running=running, horizon=self.max_steps, discount=1.0) 

		return rddl


# ----------------------------------------------------------
import networkx as nx 
import matplotlib.pyplot as plt 
   
COLOR_RUNNING = "#cad5fa"
COLOR_DOWN = "#e33c30"
COLOR_SELECTED_R = "#1b3eb5"
COLOR_SELECTED_D = "#701812"

class GraphVisualization:
	def __init__(self, env):
		self.connections = env.connections.T

		self.G = nx.DiGraph() 
		self.G.add_edges_from(self.connections) 
		self.pos = nx.kamada_kawai_layout(self.G)
		# self.pos = nx.spring_layout(self.G)

		self.colors = [COLOR_DOWN, COLOR_RUNNING, COLOR_SELECTED_D, COLOR_SELECTED_R]

		self.update_state(env)

	def update_state(self, env, a=None, probs=None):
		states = env.running.copy()
		if (a is not None):
			states[a] += 2

		self.edge_colors = np.array([self.colors[int(x)] for x in states])
		self.edge_colors = self.edge_colors[self.G.nodes]	# re-order

		if probs is not None:
			self.node_labels = {i: f"{probs[i]:.1f}".lstrip("0") for i in self.G.nodes}

			self.node_colors = np.array([(1-x, 1-x, 1-x) for x in probs])
			self.node_colors = self.node_colors[self.G.nodes]

		else:
			self.node_labels = None
			self.node_colors = ['w'] * len(states)


	def plot(self):
		plt.clf()
		nx.draw_networkx(self.G, pos=self.pos, labels=self.node_labels, node_color=self.node_colors, edgecolors=self.edge_colors, linewidths=3.0, arrows=True)
		return plt


# ----------------------------------------------------------
if __name__ == '__main__':
	NODES = 5

	env = SysAdminEnv(env_num_obj=NODES, env_max_steps=10)
	s = env.reset()

	gvis = GraphVisualization(env)

	a = -1
	while(True):
		# a = np.random.randint(env.num_obj)
		a = np.random.choice(NODES, np.random.randint(0, NODES), replace=False)
		probs = np.random.rand(NODES)

		print(a)
		print(probs)

		gvis.update_state(env, a, probs)
		gvis.plot().show()

		s, r, d, i = env.step(a)

		print(a, r)

		if d:
			gvis = GraphVisualization(env)
