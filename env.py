from utils import click, move_to_coords, get_shot, get_action_shape
import torch


class Environment:
	def __init__(self, state_size):
		self.state_size = state_size
		self.action_shape = get_action_shape()
	def step(self, actions):
		action_x, action_y, action_c = actions
		move_to_coords((action_x[0].item(), action_y[0].item()))
		click(action_c[0].item() > .5)
		return get_shot(self.state_size)
	def reset(self):
		return get_shot(self.state_size)