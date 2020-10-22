import torch

class Activation(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.swish = lambda x: x * torch.sigmoid(x)
		self.linear = lambda x: x
		self.sigmoid = lambda x: torch.sigmoid(x)
		self.neg = lambda x: -x
		self.sine = lambda x: torch.sin(x)
		
		self.params = torch.nn.Parameter(torch.zeros(10))

	def forward(self, x):
		params = torch.sigmoid(self.params)
		
		linear_x = self.linear(x) * params[0]
		swish_x = self.swish(x) * params[1]
		sigmoid_x = self.sigmoid(x) * params[2]
		neg_x = self.neg(x) * params[3]
		sine_x = self.sine(x) * params[4]

		x = swish_x + linear_x + sigmoid_x + neg_x + sine_x
		
		return x

class ResizableConv2d(torch.nn.Module):
	def __init__(self, state_size, inchan, outchan):
		super().__init__()
		self.conv = torch.nn.Conv2d(inchan, outchan, 3)
		self.conv2 = torch.nn.Conv2d(outchan, outchan, 3)
		self.residual_conv = torch.nn.Conv2d(inchan, outchan, 3)
		self.resize = lambda x: torch.nn.functional.interpolate(x, size=state_size, mode='bicubic', align_corners=True)
		self.activation = Activation()
	def forward(self, x):
		y = self.conv(x)
		y = self.conv2(y)
		y = self.resize(y)

		y = y + self.resize(self.residual_conv(x))
		y = self.activation(y)
		return y

class ActorNet(torch.nn.Module):
	def __init__(self, state_size, action_size):
		super().__init__()
		self.conv = ResizableConv2d(state_size, 6, 3)
		self.conv_backwards = ResizableConv2d(state_size, 3, 6)

		self.conv2 = ResizableConv2d(state_size, 3, 3)
		self.conv3 = ResizableConv2d(state_size, 3, 3)
		self.conv4 = ResizableConv2d(state_size, 3, 3)
		self.conv_resize = ResizableConv2d((8, 8), 3, 3)

		self.linears = torch.nn.ModuleList([])

		for i in action_size:
			self.linears.append(torch.nn.Linear(8*8*3, i))

		self.optim = torch.optim.AdamW(self.parameters(), lr=1e-4)

	def forward(self, x, goal, time):
		x = torch.cat([x, goal], dim=1) + time

		x = self.conv(x)
		x_ = self.conv_backwards(x)

		x = self.conv(x_) + goal
		x = x + torch.randn_like(x)
		x = self.conv2(x) + time
		x = x + torch.randn_like(x)
		x = self.conv3(x) + goal
		x = x + torch.randn_like(x)
		x = self.conv4(x) + goal

		x = self.conv_resize(x)

		y = x

		y = torch.flatten(y, start_dim=1)

		y_list = []
		for i in self.linears:
			iy = i(y)
			iy = torch.sigmoid(iy)	
			y_list.append(iy)

		return y_list
	
	def optimize(self, loss):
		self.optim.zero_grad()
		loss.backward()
		self.optim.step()
		print("Actor Loss:", loss.item())

class GoalkeeperNet(torch.nn.Module):
	def __init__(self, state_size):
		super().__init__()
		self.conv = ResizableConv2d(state_size, 3, 3)
		self.conv2 = ResizableConv2d(state_size, 3, 3)
		self.conv3 = ResizableConv2d(state_size, 3, 3)
		self.conv4 = ResizableConv2d(state_size, 3, 3)
		self.flatten = torch.nn.Flatten()
		self.optim = torch.optim.AdamW(self.parameters(), lr=1e-4)

	def forward(self, state):
		y = self.conv(state)
		goal = self.conv2(y)
		goal = self.conv3(goal)

		return goal


	def optimize(self, loss):
		self.optim.zero_grad()
		loss.backward()
		self.optim.step()
		print("Goalkeeper Loss:", loss.item())