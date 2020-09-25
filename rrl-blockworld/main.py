import numpy as np, scipy
import gym, torch, boxworld

import wandb, argparse, itertools, os

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
	parser.add_argument('-device', type=str, choices=['auto', 'cpu', 'cuda'] + cuda_devices, default='auto', help="Which device to use")
	parser.add_argument('-cpus', type=str, default='auto', help="How many CPUs to use")
	parser.add_argument('-batch', type=int, default=256, help="Size of a batch / How many CPUs to use")
	parser.add_argument('-seed', type=int, default=None, help="Random seed") # seed in multiprocessing is not implemented
	parser.add_argument('-load_model', type=str, default=None, help="Load model from this file")
	parser.add_argument('-epoch', type=int, default=1000, help="Epoch length")
	parser.add_argument('-max_epochs', type=int, default=None, help="Terminate after this many epochs")
	parser.add_argument('-mp_iterations', type=int, default=5, help="Number of message passes")
	parser.add_argument('-lr', type=float, default=3e-4, help="Initial learning rate")
	parser.add_argument('-alpha_h', type=float, default=0.5e-4, help="Initial entropy regularization constant")
	parser.add_argument('-boxes', type=int, default=5, help="Number of boxes")

	parser.add_argument('-trace', action='store_const', const=True, help="Show trace of the agent")
	parser.add_argument('-eval', action='store_const', const=True, help="Evaluate the agent")

	cmd_args = parser.parse_args()

	return cmd_args

# ----------------------------------------------------------------------------------------
def evaluate(net, planner):
	test_env = SubprocVecEnv([lambda: gym.make('Boxworld-v0', plan=planner) for i in range(config.eval_batch)], in_series=(config.eval_batch // config.cpus), context='fork')
	tqdm_val = tqdm(desc='Validating', total=config.eval_problems, unit=' problems')

	with torch.no_grad():
		net.eval()

		r_tot = 0.
		problems_solved = 0.
		problems_finished = 0.
		problems_timeout = 0.
		steps = 0

		opt_all = []
		opt_solved = []

		s = test_env.reset()

		while problems_finished < config.eval_problems:
			steps += 1
		# for step in range(1e9):
			a, v, pi = net(s)
			s, r, d, i = test_env.step(a)

			# print(r)
			r_tot += np.sum(r)
			problems_solved   += np.array(sum(x['d_true'] for x in i)) # conversion to numpy for easier ZeroDivision handling (-> nan)
			problems_finished += np.sum(d)

			if planner is not None:
				# print([x['path_len'] / x['steps'] if x['d_true'] else 0. for x in i if x['done']])
				opt_all += [x['path_len'] / x['steps'] if x['d_true'] else 0. for x in i if x['done']]
				opt_solved += [x['path_len'] / x['steps'] for x in i if x['d_true']]

			tqdm_val.update(np.sum(d))

		problems_solved_ps  = problems_solved / (steps * config.eval_batch)
		problems_solved_avg = problems_solved / problems_finished

		r_avg_ps = r_tot / (steps * config.eval_batch) # average reward per step
		r_avg_pp = r_tot / problems_finished # average reward per problem

		opt_all_avg = np.mean(opt_all)
		opt_all_sem = scipy.stats.sem(opt_all)

		opt_solved_avg = np.mean(opt_solved)
		opt_solved_sem = scipy.stats.sem(opt_solved)
		
		avg_steps_to_solve = (steps * config.eval_batch) / problems_finished

		net.train()

	tqdm_val.close()
	test_env.close()

	eval_log = {
		'reward_per_step': r_avg_ps,
		'reward_per_problem': r_avg_pp,
		'problems_solved': problems_solved_avg,
		'problems_finished': problems_finished,
		'solved_per_step': problems_solved_ps,
		'steps_per_problem': avg_steps_to_solve,
		'optimality_all': opt_all_avg,
		'optimality_all_sem': opt_all_sem,
		'optimality_solved': opt_solved_avg,
		'optimality_solved_sem': opt_solved_sem,
	}

	return eval_log

# ----------------------------------------------------------------------------------------
def trace_net(net, net_name, planner):
	test_env = gym.make('Boxworld-v0', plan=planner)

	with torch.no_grad():
		net.eval()
		s = test_env.reset()

		while True:
			print(f"{boxworld._get_state_string(test_env.state)} -> {boxworld._get_state_string(test_env.goal)}")

			while(True):
				a, v, pi = net([s])
				s, r, d, i = test_env.step(a[0])

				print(f"move{a[0]}: {boxworld._get_state_string(i['raw_state'])}")

				if d: break;

			print("optimal" if i['steps'] == i['path_len'] else "not optimal")
			input()

			# print(f"\t\t\t{a=} {r=}, {d=}, {i=}")
			# print(f"\t\t\t{a=} {r=}, {d=}, opt={i['path_len']}")

		net.train()

# ----------------------------------------------------------------------------------------
if __name__ == '__main__':
	args = get_args()
	config.init(args)

	np.set_printoptions(threshold=9999)
	torch.set_printoptions(sci_mode=True)

	print(f"Config: {config}")

	gym.envs.registration.register(
		id='Boxworld-v0',
		entry_point='boxworld:BoxworldEnv',
		kwargs={'box_max_steps': config.box_max_steps, 'box_num_obj': config.box_num_obj}
	)

	if config.seed:
		init_seed(config.seed)

	torch.set_num_threads(config.cpus)	
	
	net = Net()
	target_net = Net()

	if config.load_model:
		net.load(config.load_model)
		target_net.load(config.load_model)

		print(f"Model loaded: {config.load_model}")

	# --- decide what planner to use
	if config.box_num_obj <= 9:	# use internal a-star implementation
		planner = 'astar'
	# elif config.box_num_obj <= 10: # use external planner
	# 	planner = 'external'
	else:
		planner = None
	# ---

	if args.trace:
		trace_net(net, config.load_model, planner)
		exit(0)

	if args.eval:
		print( evaluate(net, planner) )
		exit(0)

	env = SubprocVecEnv([lambda: gym.make('Boxworld-v0') for i in range(config.batch)], in_series=(config.batch // config.cpus), context='fork')

	# job_name = f"{config.soko_size[0]}x{config.soko_size[1]}-{config.soko_boxes} mp-{config.mp_iterations} nn-{config.emb_size} b-{config.batch}"
	job_name = None 
	wandb.init(project="rrl-boxworld", name=job_name, config=config)
	wandb.save("*.pt")

	wandb.watch(net, log='all')
	# print(net)

	tot_env_steps = 0
	tot_el_env_steps = 0

	tqdm_main = tqdm(desc='Training', unit=' steps')
	s = env.reset()

	for step in itertools.count(start=1):
		a, v, pi = net(s)
		s, r, d, i = env.step(a)
		# print(r, d)
		# print(s)

		s_true = [x['s_true'] for x in i]
		d_true = [x['d_true'] for x in i]

		n_stacks = list(len(x['raw_state']) for x in i)	# for the entropy regularization

		# update network
		loss, loss_pi, loss_v, loss_h, entropy, norm = net.update(r, v, pi, s_true, n_stacks, d_true, target_net)
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

			eval_log = evaluate(net, planner)
			# debug_net(net)

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
			wandb.log(log, commit=False)
			wandb.log(eval_log)

			# save model to wandb
			net.save(os.path.join(wandb.run.dir, "model.pt"))

		# finish if max_epochs exceeded
		if config.max_epochs and (step // config.log_rate >= config.max_epochs):
			break

	env.close()
	tqdm_main.close()