from env import Environment
from agent import Agent
import time
import numpy as np

def act_and_store(s):
	action = agent.act(s)
	ns = env.step(action)
	agent.remember(s, action, ns)
	return ns

if __name__ == '__main__':
	env = Environment((300,300))
	time.sleep(10)
	agent = Agent((300,300), env.action_shape)
	s = env.reset()
	s = act_and_store(s)
	while True:
		func = np.random.choice([agent.replay_actor, agent.replay_goalkeeper])
		s = act_and_store(s)
		func()



