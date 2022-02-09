import numpy as np
import time
from common import get_args,experiment_setup
import tensorflow as tf

if __name__=='__main__':

	tf.compat.v1.disable_eager_execution()
	# After eager execution is enabled, operations are executed as they are
	# defined and Tensor objects hold concrete values, which can be accessed as
	# numpy.ndarray`s through the numpy() method.

	args = get_args()
	env, env_test, agent, buffer, learner, tester = experiment_setup(args)

	args.logger.summary_init(agent.graph, agent.sess)

	# Progress info
	args.logger.add_item('Epoch')
	args.logger.add_item('Cycle')
	args.logger.add_item('Episodes@green')
	args.logger.add_item('Timesteps')
	args.logger.add_item('TimeCost(sec)')

	# Algorithm info
	for key in agent.train_info.keys():
		args.logger.add_item(key, 'scalar')

	# Test info
	for key in tester.info:
		args.logger.add_item(key, 'scalar')

	args.logger.summary_setup()

	for epoch in range(args.epochs):
		for cycle in range(args.cycles):
			args.logger.tabular_clear()
			args.logger.summary_clear()
			start_time = time.time()

			learner.learn(args, env, env_test, agent, buffer)
			tester.cycle_summary()

			args.logger.add_record('Epoch', str(epoch)+'/'+str(args.epochs))
			args.logger.add_record('Cycle', str(cycle)+'/'+str(args.cycles))
			args.logger.add_record('Episodes', buffer.counter)
			args.logger.add_record('Timesteps', buffer.steps_counter)
			args.logger.add_record('TimeCost(sec)', time.time()-start_time)

			args.logger.tabular_show(args.tag)
			args.logger.summary_show(buffer.counter)

		tester.epoch_summary()
		# Save periodic policy every epoch
		policy_file = args.logger.my_log_dir + "saved_policy"
		agent.saver.save(agent.sess, policy_file, global_step=epoch)
		args.logger.info("Saved periodic policy to {}!".format(args.logger.my_log_dir))

	tester.final_summary()
