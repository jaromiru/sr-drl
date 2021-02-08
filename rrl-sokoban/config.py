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
		self.gamma = 0.99
		self.batch = args.batch

		self.subset = args.subset
		self.pos_feats = args.pos_feats

		self.epoch = args.epoch

		self.alpha_v = 0.1
		self.alpha_h = 0.2

		self.target_rho = 0.005
		self.emb_size = 64
		self.mp_iterations = args.mp_iterations

		self.seed = args.seed
		self.device = get_device(args.device)
		self.cpus = get_cpu_count(args.cpus)

		self.opt_lr = 3e-3
		self.opt_l2 = 1.0e-4
		self.opt_max_norm = 5.0

		self.sched_lr_factor = 0.5
		self.sched_lr_min    = 1.0e-4
		self.sched_lr_rate   = 20 * self.epoch
		
		self.sched_alpha_h_factor = 1.0
		self.sched_alpha_h_min    = 0.1
		self.sched_alpha_h_rate   = 1 * self.epoch


		self.soko_max_steps = 200

		self.q_range = (-15, 15)	# range of the Q function to help the optimization, can be None

		self.max_epochs = args.max_epochs
		self.log_rate = 1 * self.epoch
		self.eval_problems = 1000
		# self.eval_steps = 500
		self.eval_batch = 64

		self.load_model = args.load_model

		if args.custom:
			x, y, boxes = map(int, args.custom.split('x'))

			self.boxoban = False
			self.soko_size = (x, y)
			self.soko_boxes = boxes

		else:
			self.boxoban = True
			self.soko_size = (10, 10)
			self.soko_boxes = 4

	def __str__(self):
		return str( vars(self) )

config = Object()


