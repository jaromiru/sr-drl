import numpy as np, scipy.stats
import gym, torch, sysadmin

import wandb, argparse, itertools, os, glob, json

from vec_env.subproc_vec_env import SubprocVecEnv
from net import Net
from tqdm import tqdm

from config import config

# ----------------------------------------------------------------------------------------
def decay_time(step, start, min, factor, rate):
	exp = step / rate * factor
	value = (start - min) / (1 + exp) + min

	return value

def decay_exp(step, start, min, factor, rate):
	exp = step / rate
	value = (start - min) * (factor ** exp) + min

	return value

def init_seed(seed):
	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)

def get_args():
	cuda_devices = [f'cuda:{i}' for i in range(torch.cuda.device_count())]

	parser = argparse.ArgumentParser()
	parser.add_argument('-device', type=str, choices=['auto', 'cpu', 'cuda'] + cuda_devices, default='cpu', help="Which device to use")
	parser.add_argument('-cpus', type=str, default=1, help="How many CPUs to use (use 'auto' for all available cpus)")
	parser.add_argument('-batch', type=int, default=256, help="Size of a batch / How many CPUs to use")
	parser.add_argument('-seed', type=int, default=None, help="Random seed") # seed in multiprocessing is not implemented
	parser.add_argument('-load_model', type=str, default=None, help="Load model from this file")
	parser.add_argument('-epoch', type=int, default=100, help="Epoch length")
	parser.add_argument('-max_epochs', type=int, default=50, help="Terminate after this many epochs")

	parser.add_argument('--multi', action='store_const', const=True, help="Allow concurrent actions")
	
	parser.add_argument('-mp_iterations', type=int, default=5, help="Number of message passes")
	parser.add_argument('-lr', type=float, default=3e-3, help="Initial learning rate")
	parser.add_argument('-alpha_h', type=float, default=0.3, help="Initial entropy regularization constant")
	
	parser.add_argument('-nodes', type=int, default=5, help="Number of nodes")

	parser.add_argument('-trace', action='store_const', const=True, help="Show trace of the agent")
	parser.add_argument('-eval', action='store_const', const=True, help="Evaluate the agent")
	parser.add_argument('--save_domains', action='store_const', const=True, help="Save the evaluated RDDL domains")
	parser.add_argument('--baseline', type=str, choices=['random', 'random_down'], default=None, help="Evaluate baseline")
	parser.add_argument('--print_raw', action='store_const', const=True, help="Print mean reward with 95ci")
	parser.add_argument('--eval_problems', type=int, default=100, help="Number of problems to evaluate")

	cmd_args = parser.parse_args()

	return cmd_args

# ----------------------------------------------------------------------------------------
def random_action(s, baseline, multi):
	if baseline == 'random':
		if multi:
			a = [np.random.choice(config.env_num_obj, np.random.randint(0, config.env_num_obj), replace=False) for x in s]
		else:
			a = np.random.randint(0, config.env_num_obj, size=(config.eval_batch, 1))

	elif baseline == 'random_down':
		a = []
		for state in s:
			running = state[0]
			down_indices = np.flatnonzero(running == 0)

			if down_indices.size == 0:
				a_ = []
			else:
				if multi:
					a_ = down_indices
				else:
					a_ = [np.random.choice(down_indices)]

			a.append(a_)

	return a

# ----------------------------------------------------------------------------------------
def evaluate(net, save_domains=False, baseline=None):
	test_env = SubprocVecEnv([lambda: gym.make('SysAdmin-v0', save_domain=save_domains) for i in range(config.eval_batch)], in_series=(config.eval_batch // config.cpus), context='fork')
	tqdm_val = tqdm(desc='Validating', total=config.eval_problems, unit=' problems')

	with torch.no_grad():
		net.eval()

		r_tot = 0.
		problems_finished = 0.
		rewards = []
		steps = 0

		s = test_env.reset()

		while problems_finished < config.eval_problems:
			steps += 1

			if not baseline:
				a, v, pi, pi_full = net(s)
			else:
				a = random_action(s, baseline, config.multi)

			s, r, d, i = test_env.step(a)

			r_tot += np.sum(r)
			problems_finished += np.sum(d)
			rewards += [x['reward_total'] for x  in itertools.compress(i, d)]

			tqdm_val.update(np.sum(d))

		r_avg_ps = r_tot / (steps * config.eval_batch) # average reward per step
		r_avg_pp = r_tot / problems_finished # average reward per problem

		net.train()

	if args.print_raw:
		rew_mean = np.mean(rewards)
		rew_ci95 = 1.96 * scipy.stats.sem(rewards)
		print(f"{rew_mean:.2f} ± {rew_ci95:.2f}")

	tqdm_val.close()
	test_env.close()

	eval_log = {
		'reward_per_step': r_avg_ps,
		'reward_per_problem': r_avg_pp,
		'rewards': rewards,
		'problems_finished': problems_finished,
	}

	return eval_log

# ----------------------------------------------------------------------------------------
def debug_net(net):
	test_env = gym.make('SysAdmin-v0')

	with torch.no_grad():
		net.eval()

		plots = []

		s = test_env.reset()
		gvis = sysadmin.GraphVisualization(test_env)

		for i in range(10):
			a, v, pi, pi_full = net([s])
			a = a[0] # only single env

			gvis.update_state(test_env, a, pi_full)
			plot = wandb.Image(gvis.plot())
			plots.append(plot)

			s, r, d, i = test_env.step(a)

		net.train()

	test_env.close()

	debug_log = {
		'plots': plots
	}

	return debug_log

# ----------------------------------------------------------------------------------------
# def trace_net(net, net_name, planner):
# 	test_env = gym.make('Boxworld-v0', plan=planner)

# 	with torch.no_grad():
# 		net.eval()
# 		s = test_env.reset()

# 		while True:
# 			print(f"{boxworld._get_state_string(test_env.state)} -> {boxworld._get_state_string(test_env.goal)}")

# 			while(True):
# 				a, v, pi = net([s])
# 				s, r, d, i = test_env.step(a[0])

# 				print(f"move{a[0]}: {boxworld._get_state_string(i['raw_state'])}")

# 				if d: break;

# 			print("optimal" if i['steps'] == i['path_len'] else "not optimal")
# 			input()

# 			# print(f"\t\t\t{a=} {r=}, {d=}, {i=}")
# 			# print(f"\t\t\t{a=} {r=}, {d=}, opt={i['path_len']}")

# 		net.train()

# ----------------------------------------------------------------------------------------
if __name__ == '__main__':
	args = get_args()
	config.init(args)

	np.set_printoptions(threshold=9999)
	torch.set_printoptions(sci_mode=True)

	print(f"Config: {config}")

	gym.envs.registration.register(
		id='SysAdmin-v0',
		entry_point='sysadmin:SysAdminEnv',
		kwargs={'env_max_steps': config.env_max_steps, 'env_num_obj': config.env_num_obj, 'multi': config.multi}
	)

	if config.seed:
		init_seed(config.seed)

	torch.set_num_threads(config.cpus)	
	
	net = Net(config.multi)
	target_net = Net(config.multi)

	if config.load_model:
		net.load(config.load_model)
		target_net.load(config.load_model)

		print(f"Model loaded: {config.load_model}")

	# if args.trace:
	# 	trace_net(net, config.load_model, planner)
	# 	exit(0)

	if args.eval:
		if args.save_domains:
			for fl in glob.glob("_plan/sysadmin_inst_*.rddl"):
			    os.remove(fl)

		print( json.dumps(evaluate(net, save_domains=args.save_domains, baseline=args.baseline)) )
		exit(0)

	env = SubprocVecEnv([lambda: gym.make('SysAdmin-v0', offset=i) for i in range(config.batch)], in_series=(config.batch // config.cpus), context='fork')

	variation = "M" if config.multi else "S"
	job_name = f"{variation} {config.env_num_obj} mp{config.mp_iterations} ah={config.alpha_h} lr={config.opt_lr}"
	wandb.init(project="rrl-sysadmin", name=job_name, config=config)
	wandb.save("*.pt")

	wandb.watch(net, log='all')
	# print(net)

	tot_env_steps = 0
	tot_el_env_steps = 0

	tqdm_main = tqdm(desc='Training', unit=' steps')
	s = env.reset()

	for step in itertools.count(start=1):
		a, v, pi, pi_full = net(s)
		s, r, d, i = env.step(a)
		# print(r, d)
		# print(s)

		s_true = [x['s_true'] for x in i]
		d_true = [x['d_true'] for x in i]

		num_obj = [x['num_obj'] for x in i]

		# update network
		loss, loss_pi, loss_v, loss_h, entropy, norm = net.update(r, v, pi, s_true, num_obj, d_true, target_net)
		target_net.copy_weights(net, rho=config.target_rho)

		# save step stats
		tot_env_steps += config.batch

		tqdm_main.update()

		if step % config.sched_lr_rate == 0:
			lr = decay_exp(step, config.opt_lr, config.sched_lr_min, config.sched_lr_factor, config.sched_lr_rate)
			net.set_lr(lr)

		if step % config.sched_alpha_h_rate == 0:
			alpha_h = decay_time(step, config.alpha_h, config.sched_alpha_h_min, config.sched_alpha_h_factor, config.sched_alpha_h_rate)
			net.set_alpha_h(alpha_h)

		if step % config.log_rate == 0:
			log_step = step // config.log_rate

			eval_log = evaluate(net)
			debug_log = debug_net(net)

			log = {
				'env_steps': tot_env_steps,

				'rate': tqdm_main.format_dict['rate'],
				'loss': loss,
				'loss_pi': loss_pi,
				'loss_v': loss_v,
				'loss_h': loss_h,
				'entropy estimate': entropy,
				'gradient norm': norm,
				'value': v.mean(),

				'lr': net.lr,
				'alpha_h': net.alpha_h,
			}

			print(log, eval_log)

			wandb.log(eval_log, commit=False)
			wandb.log(debug_log, commit=False)
			wandb.log(log)

			# save model to wandb
			net.save(os.path.join(wandb.run.dir, "model.pt"))

		# finish if max_epochs exceeded
		if config.max_epochs and (step // config.log_rate >= config.max_epochs):
			break

	env.close()
	tqdm_main.close()