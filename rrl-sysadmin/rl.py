import torch

# reward(s,a), value(s), value(s_), pi(a|s)
def a2c(r, v, v_, pi, gamma, alpha_v, alpha_h, q_range=None, log_num_actions=None):
	# make sure all dimensions are correct
	r = r.flatten()
	v = v.flatten()
	v_ = v_.flatten()
	pi = pi.flatten()

	# compute losses
	log_pi = torch.log(pi + 1e-9)	# bug fix in torch.multinomial: this should never be zero...
	q = r + gamma * v_.detach()

	if q_range is not None:
		v_target = q.clamp(*q_range)
	else:
		v_target = q

	adv = q - v
	v_err = v_target - v

	loss_pi = -adv.detach() * log_pi
	loss_v  = v_err ** 2					# can use experience replay here

	# print()
	# print(r)
	# print(v)
	# print(v)
	# print(pi)
	# exit()

	# print(adv)
	# print(log_pi)
	# print(loss_pi)
	# print(loss_v)
	# exit()

	if log_num_actions is not None:
		log_num_actions = log_num_actions.flatten()
		# legal_actions = num_actions > 1
		# num_actions[~legal_actions] = 2	# bug fix in pytorch: to avoid nan error during backprop

		loss_h = (log_pi.detach() * log_pi) / log_num_actions # scale the entropy with its maximum
		ent = log_pi / log_num_actions
		entropy = -torch.mean(ent)      # normalized entropy estimate, for logging purposes

		# entropy = -torch.mean(ent[legal_actions])      # normalized entropy estimate, for logging purposes
	else:
		loss_h  = log_pi.detach() * log_pi
		entropy = -torch.mean(log_pi)

	loss_pi  = torch.mean(loss_pi)
	loss_v   = alpha_v * torch.mean(loss_v)
	loss_h   = alpha_h * torch.mean(loss_h)

	loss = loss_pi + loss_v + loss_h

	# print(f"{loss.item()=} {loss_pi=} {loss_v=} {loss_h=}")
	return loss, loss_pi, loss_v, loss_h, entropy