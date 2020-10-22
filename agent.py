import torch

from net import GoalkeeperNet, ActorNet
from memory import MemoryBank

class Agent:
	def __init__(self, state_size, action_size, mem_size=100):
		self.actor = ActorNet(state_size, action_size)
		self.goalkeeper = GoalkeeperNet(state_size)
		self.memory = MemoryBank(maxlen=mem_size)
		self.timestep = torch.Tensor([[0.0]])
		self.eps = 1.0
		self.eps_decay = 0.994
		self.eps_min = 0.001

	def act(self, state):
		with torch.no_grad():
			goal = self.goalkeeper(state)
			action = self.actor(state, goal, torch.Tensor([[0.2]]))
			self.timestep += 0.001
		return action

	def remember(self, state, action, next_state):
		self.memory.append((state, action, next_state, self.timestep))

	def replay_actor(self):
		for (rs, ra, rns, rt), (ls, la, lns, lt) in zip(*self.memory.sample_in_order()):
			ds = (rs + ls)
			dt = lt - rt
			
			loss = 0.0
			ra_pred = self.actor(rs, ds, dt)
			for ract_pred, ract in zip(ra_pred, ra):
				loss += torch.nn.functional.mse_loss(ract_pred, ract.detach())

			self.actor.optimize(loss)

	def replay_goalkeeper(self):
		for (rs, ra, rns, rt), (ls, la, lns, lt) in zip(*self.memory.sample_in_order()):
			goal = self.goalkeeper(rs)

			loss = 0.0
			ra_pred = self.actor(rs, goal, torch.Tensor([[0.3]]))
			for ract_pred, ract in zip(ra_pred, ra):
				loss += torch.nn.functional.mse_loss(ract_pred, ract.detach())
			loss = (-loss)

			self.goalkeeper.optimize(loss)