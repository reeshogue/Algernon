import numpy as np

class MemoryBank(object):
	def __init__(self, maxlen=100):
		self.memories = []
		self.maxlen = maxlen
	def append(self, memory):
		self.memories.append(memory)
		if len(self.memories) > self.maxlen:
			self.memories.pop(-1)

	def sample_in_order(self):
		memory_one_idx = np.random.choice(len(self.memories))
		memory_two_idx = np.random.choice(len(self.memories))
		
		if memory_one_idx < memory_two_idx or memory_two_idx < memory_one_idx:
			memory_two_idx, memory_one_idx = memory_one_idx, memory_two_idx

		memory_one_access = [self.memories[memory_one_idx]]
		memory_two_access = [self.memories[memory_two_idx]]
		
		return memory_one_access, memory_two_access