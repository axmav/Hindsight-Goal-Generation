from .ddpg2 import DDPG

def create_agent(args):
	return {
		'ddpg': DDPG
	}[args.alg](args)