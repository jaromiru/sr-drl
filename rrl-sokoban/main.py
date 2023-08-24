import numpy as np
import gym, gym_sokoban, torch

import wandb, argparse, itertools, os

from vec_env.subproc_vec_env import SubprocVecEnv
from net import Net
from tqdm import tqdm

from config import config

# ----------------------------------------------------------------------------------------
def to_action(a, n, s, size):
	node_indices = [x[4] for x in s]

	a = a.cpu().numpy()
	n = n.cpu().numpy()

	nodes = [indices[n[i]] for i, indices in enumerate(node_indices)]

	actions = [ ((nodes[i][1], nodes[i][0]), a[i]) for i in range(len(a)) ] # requires ( (x, y), action )
	return actions

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

	# optimal cpu=2, device=cuda (rate 3.5)
	parser = argparse.ArgumentParser()
	parser.add_argument('-device', type=str, choices=['auto', 'cpu', 'cuda'] + cuda_devices, default='cpu', help="Which device to use")
	parser.add_argument('-cpus', type=str, default='2', help="How many CPUs to use")
	parser.add_argument('-batch', type=int, default=256, help="Size of a batch")
	parser.add_argument('-seed', type=int, default=None, help="Random seed") # seed in multiprocessing is not implemented
	parser.add_argument('-load_model', type=str, default=None, help="Load model from this file")
	parser.add_argument('-resume_id', type=str, default=None, help="Resume this wandb id")
	parser.add_argument('-resume_step', type=int, default=1, help="Start with this step")
	parser.add_argument('-epoch', type=int, default=1000, help="Epoch length")
	parser.add_argument('-max_epochs', type=int, default=None, help="Terminate after this many epochs")

	parser.add_argument('-mp_iterations', type=int, default=10, help="Number of message passes")
	# parser.add_argument('-att_heads', type=int, default=1, help="Number of attention heads")
	parser.add_argument('-emb_size', type=int, default=64, help="Embedding size")

	parser.add_argument('-lr', type=float, default=3e-3, help="Initial learning rate")
	parser.add_argument('-alpha_h', type=float, default=0.04, help="Initial entropy regularization constant")
	parser.add_argument('-max_norm', type=float, default=10., help="Maximal gradient norm")

	parser.add_argument('-precond', action='store_const', const=True, help="Use preconditions")
	parser.add_argument('-subset', type=int, default=None, help="Use a subset of train set")
	parser.add_argument('-pos_feats', action='store_const', const=True, help="Enable positional features")
	parser.add_argument('--custom', type=str, default=None, help="Custom size (e.g. 10x10x4; else Boxoban)")

	parser.add_argument('-trace', action='store_const', const=True, help="Show trace of the agent")
	parser.add_argument('-eval', action='store_const', const=True, help="Evaluate the agent")

	cmd_args = parser.parse_args()

	return cmd_args

# ----------------------------------------------------------------------------------------
def evaluate(net, split='valid', subset=None):
	test_env = SubprocVecEnv([lambda: gym.make('Sokograph-v0', split=split, subset=subset) for i in range(config.eval_batch)], in_series=(config.eval_batch // config.cpus), context='fork')
	tqdm_val = tqdm(desc=f'Evaluation ({split})', total=config.eval_problems, unit=' problems')

	with torch.no_grad():
		net.eval()

		r_tot = 0.
		problems_solved = 0
		problems_finished = 0
		steps = 0
		terminated = np.zeros(config.eval_batch, dtype=bool)

		s = test_env.reset()

		while True:
			steps += np.sum(~terminated)

			a, n, v, pi = net(s)
			actions = to_action(a, n, s, size=config.soko_size)

			s, r, d, i = test_env.step(actions)

			r_tot += np.sum( r * ~terminated )
			problems_solved += sum('all_boxes_on_target' in i[idx] and i[idx]['all_boxes_on_target'] == True for idx in np.nonzero(~terminated)[0])
			problems_finished += np.sum(d * ~terminated)
			# print(d * ~terminated)

			tqdm_val.update(np.sum(d * ~terminated))

			if problems_finished >= config.eval_problems:
				terminated |= d
				if np.all(terminated):
					break

		r_avg = r_tot / steps # average reward per step
		problems_solved_ps  = problems_solved / steps
		problems_solved_avg = problems_solved / problems_finished

		net.train()

	tqdm_val.close()
	test_env.close()

	return r_avg, problems_solved_ps, problems_solved_avg, problems_finished

# ----------------------------------------------------------------------------------------
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def get_plotly(net, env, s, title=None):
	s_img = env.render(mode='rgb_array')

	action_softmax, node_softmaxes, value = net([s], complete=True)

	action_probs = action_softmax.probs[0].cpu()
	# node_probs = node_softmaxes[0].reshape(*config.soko_size, 5).flip(0).cpu() # flip is because Heatmap flips the display :-(
	value = value[0].item()

	node_indices = s[4]
	node_probs = np.zeros((*config.soko_size, 4))
	node_probs[tuple(node_indices.T)] = node_softmaxes[0].cpu()
	node_probs = np.flip(node_probs, 0)

	fig = make_subplots(rows=2, cols=4, subplot_titles=["State", "", "PUSH_UP", "State value", "Action probs", "PUSH_LEFT", "PUSH_DOWN", "PUSH_RIGHT"])

	fig.add_trace(go.Image(z=s_img), 1, 1)
	fig.add_trace(go.Bar(x=["PUSH_UP", "PUSH_DOWN", "PUSH_LEFT", "PUSH_RIGHT"], y=action_probs), 2, 1)
	fig.add_trace(go.Bar(x=["value"], y=[value], text=[f'{value:.2f}'], textposition='auto', width=[0.2]), 1, 4)
	fig.update_yaxes(range=config.q_range, row=1, col=4)

	# fig.add_trace(go.Heatmap(z=node_probs[:, :, 0], zmin=0., zmax=1., colorscale='Greys', showscale=False), 1, 2)	# MOVE_TO
	fig.add_trace(go.Heatmap(z=node_probs[:, :, 0], zmin=0., zmax=1., colorscale='Greys', showscale=False), 1, 3)	# PUSH_UP
	fig.add_trace(go.Heatmap(z=node_probs[:, :, 1], zmin=0., zmax=1., colorscale='Greys', showscale=False), 2, 3) # PUSH_DOWN
	fig.add_trace(go.Heatmap(z=node_probs[:, :, 2], zmin=0., zmax=1., colorscale='Greys', showscale=False), 2, 2) # PUSH_LEFT
	fig.add_trace(go.Heatmap(z=node_probs[:, :, 3], zmin=0., zmax=1., colorscale='Greys', showscale=True),  2, 4) # PUSH_RIGHT

	fig.update_layout(showlegend=False, title=title, title_x=0.5)

	return fig, value, action_probs

def debug_net(net):
	test_env = gym.make('Sokograph-v0', split='valid')
	s = test_env.reset()
	# s_img = test_env.render(mode='rgb_array')

	with torch.no_grad():
		net.eval()
		fig, value, action_probs = get_plotly(net, test_env, s)
		net.train()

	wandb.log({'net_debug': fig, 'value': value, 
		'aprob_up': action_probs[0], 'aprob_down': action_probs[1], 
		'aprob_left': action_probs[2], 'aprob_right': action_probs[3]}, commit=False)

def trace_net(net, net_name, steps=10):
	import imageio, io
	from pdfrw import PdfReader, PdfWriter

	test_env = gym.make('Sokograph-v0', split='valid')
	s = test_env.reset()

	with torch.no_grad():
		net.eval()
		imgs = []

		tqdm_trace = tqdm(desc='Creating trace', unit=' steps', total=steps)
		for step in range(steps):
			fig, _, _ = get_plotly(net, test_env, s, title=f"{net_name} | step {test_env.step_idx} ({test_env.env.env.num_env_steps})")
			imgs.append(fig.to_image(format='pdf'))

			# make a regular step
			a, n, v, pi = net([s])
			actions = to_action(a, n, [s], size=config.soko_size)
			s, r, d, i = test_env.step(actions[0])

			tqdm_trace.update()

		writer = PdfWriter()
		for img in imgs:
			pdf_img = PdfReader(io.BytesIO(img)).pages
			writer.addpages(pdf_img)

		writer.write('trace.pdf')

		net.train()

# ----------------------------------------------------------------------------------------
if __name__ == '__main__':
	args = get_args()
	config.init(args)

	print(f"Config: {config}")

	gym.envs.registration.register(
		id='Sokograph-v0',
		entry_point='graph_sokoban:GraphSokoban',
		kwargs={'difficulty': 'unfiltered', 'max_steps': config.soko_max_steps, 'pos_feats': config.pos_feats, 
			'boxoban': config.boxoban, 'soko_size': config.soko_size, 'soko_boxes': config.soko_boxes}
	)

	if config.seed:
		init_seed(config.seed)

	torch.set_num_threads(config.cpus)	
	
	net = Net()
	target_net = Net()
	print(net)

	if config.load_model:
		net.load(config.load_model)
		target_net.load(config.load_model)

		print(f"Model loaded: {config.load_model}")

	if args.trace:
		trace_net(net, config.load_model)
		exit(0)

	if args.eval:
		r_avg, s_ps_avg, s_avg, s_tot = evaluate(net)
		print(f"Avg. reward: {r_avg}, Avg. solved per step: {s_ps_avg}, Avg. solved: {s_avg}, Tot. finished: {s_tot}")
		exit(0)

	env = SubprocVecEnv([lambda: gym.make('Sokograph-v0', subset=config.subset) for i in range(config.batch)], in_series=(config.batch // config.cpus), context='fork')
	# env = ParallelEnv('Sokograph-v0', n_envs=N_ENVS, cpus=N_CPUS)

	job_name = f"{config.soko_size[0]}x{config.soko_size[1]}-{config.soko_boxes} mp-{config.mp_iterations} nn-{config.emb_size} b-{config.batch}"
	
	if args.resume_id:
		wandb.init(project="rrl-sokoban", name=job_name, config=config, id=args.resume_id, resume=True)
	else:
		wandb.init(project="rrl-sokoban", name=job_name, config=config)

	wandb.save("*.pt")

	wandb.watch(net, log='all')

	tot_env_steps = 0
	tot_el_env_steps = 0

	tqdm_main = tqdm(desc='Training', unit=' steps')
	s = env.reset()

	lr = decay_exp(args.resume_step, config.opt_lr, config.sched_lr_min, config.sched_lr_factor, config.sched_lr_rate)
	net.set_lr(lr)

	alpha_h = decay_time(args.resume_step, config.alpha_h, config.sched_alpha_h_min, config.sched_alpha_h_factor, config.sched_alpha_h_rate)
	net.set_alpha_h(alpha_h)

	for step in itertools.count(start=args.resume_step):
		a, n, v, pi = net(s)
		actions = to_action(a, n, s, size=config.soko_size)

		# print(actions)
		s, r, d, i = env.step(actions)

		s_true = [x['s_true'] for x in i]
		d_true = [x['d_true'] for x in i]

		# update network
		loss, loss_pi, loss_v, loss_h, entropy, norm = net.update(r, v, pi, s_true, d_true, target_net)
		target_net.copy_weights(net, rho=config.target_rho)

		# save step stats
		tot_env_steps += config.batch
		tot_el_env_steps += np.sum([x['elementary_steps'] for x in i])

		tqdm_main.update()

		if step % config.sched_lr_rate == 0:
			lr = decay_exp(step, config.opt_lr, config.sched_lr_min, config.sched_lr_factor, config.sched_lr_rate)
			net.set_lr(lr)

		if step % config.sched_alpha_h_rate == 0:
			alpha_h = decay_time(step, config.alpha_h, config.sched_alpha_h_min, config.sched_alpha_h_factor, config.sched_alpha_h_rate)
			net.set_alpha_h(alpha_h)

		if step % (config.log_rate * 10) == 0:
			debug_net(net)

		if step % config.log_rate == 0:
			epoch = (step // config.epoch) - 1
			# log_step = (step // config.log_rate) - 1

			r_avg, s_ps_avg, s_avg, _ = evaluate(net)
			# r_avg_trn, s_ps_avg_trn, s_avg_trn, _ = evaluate(net, split='train', subset=config.subset)

			log = {
				'env_steps': tot_env_steps,
				'el_env_steps': tot_el_env_steps,
				'epoch': epoch,

				'rate': tqdm_main.format_dict['rate'],
				'loss': loss,
				'loss_pi': loss_pi,
				'loss_v': loss_v,
				'loss_h': loss_h,
				'entropy estimate': entropy,
				'gradient norm': norm,

				'lr': net.lr,
				'alpha_h': net.alpha_h,

				'reward_avg': r_avg,
				'solved_per_step': s_ps_avg,
				'solved_avg': s_avg,

				# 'reward_avg_train': r_avg_trn,
				# 'solved_per_step_train': s_ps_avg_trn,
				# 'solved_avg_train': s_avg_trn
			}

			print(log)
			wandb.log(log, step=epoch, commit=True)

			# save model to wandb
			net.save(os.path.join(wandb.run.dir, "model.pt"))

		# finish if max_epochs exceeded
		if config.max_epochs and (step // config.epoch >= config.max_epochs):
			break

	env.close()
	tqdm_main.close()