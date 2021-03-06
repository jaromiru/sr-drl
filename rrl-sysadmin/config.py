import torch, os

def get_cpu_count(cpus):
	if cpus == 'auto':
		slurm_cpus = os.environ.get('SLURM_CPUS_ON_NODE')
		node_cpus = os.cpu_count()

		if slurm_cpus is not None:
			return int(slurm_cpus)
		else:
			return node_cpus

	else:
		return int(cpus)

def get_device(device):
	if device == 'auto':
		return 'cuda' if torch.cuda.is_available() else 'cpu'
	else:
		return device


class Object:
	def init(self, args):
		self.multi = args.multi

		self.gamma = 0.99
		self.batch = args.batch

		self.epoch = args.epoch

		self.alpha_v = 0.1
		self.alpha_h = args.alpha_h

		self.target_rho = 0.005
		# self.emb_size = 32
		self.mp_iterations = args.mp_iterations

		self.seed = args.seed
		self.device = get_device(args.device)
		self.cpus = get_cpu_count(args.cpus)

		self.opt_lr = args.lr
		self.opt_l2 = 1.0e-4
		self.opt_max_norm = 3.0

		self.sched_lr_factor = 0.5
		self.sched_lr_min    = self.opt_lr / 30.
		self.sched_lr_rate   = 20 * self.epoch
		
		self.sched_alpha_h_factor = 1.0
		self.sched_alpha_h_min    = self.alpha_h / 2.
		self.sched_alpha_h_rate   = 1 * self.epoch

		self.env_num_obj = args.nodes
		self.env_max_steps = 100

		# range of the Q function to help the optimization, can be None
		self.q_range = (-100., 200. * args.nodes)	

		self.max_epochs = args.max_epochs
		self.log_rate = 1 * self.epoch
		self.eval_problems = args.eval_problems
		self.eval_batch = min(100, args.eval_problems)

		self.load_model = args.load_model

	def __str__(self):
		return str( vars(self) )

config = Object()


