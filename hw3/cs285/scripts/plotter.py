import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

def get_eval_avg_returns(filename, metric='Eval_AverageReturn'):
	eval_returns = []
	for root, dir, files in os.walk(filename):
		for e in tf.train.summary_iterator(filename + '/' + files[0]):
			for v in e.summary.value:
				if v.tag == metric:
					eval_returns.append(v.simple_value)
		break
	return eval_returns

def plot_data(logdir, head='../data/', scale=10000, title=[]):
	data = [get_eval_avg_returns(head + log) for log in logdir]
	for i, dat in enumerate(data):
		x = np.arange(1, len(dat)+1) * scale
		#plt.scatter(x, dat)
		plt.plot(x, dat)
		plt.xlabel('Number of Iterations')
		plt.ylabel('Eval_AverageReturn')
		if title is not None:
			plt.title(title[i])
		plt.show()
		plt.savefig('../data/' + logdir[i] + '.png')
		plt.clf()

def plot_data_same_graph(logdir, title, head='../data/', metric='Eval_AverageReturn', scale=10000, labels=[], legend_title=''):
	if labels == []:
		labels = logdir
	data = [get_eval_avg_returns(head + log, metric=metric) for log in logdir]
	for i, dat in enumerate(data):
		x = np.arange(1, len(dat)+1) * scale
		#plt.scatter(x, dat)
		plt.plot(x, dat, label=labels[i])
	plt.xlabel('Number of Iterations')
	plt.ylabel(metric)
	plt.legend(title=legend_title)
	plt.show()
	plt.savefig('../data/' + title + '.png')
	plt.clf()

def get_q1_data(filename):
	eval_returns = []
	best_returns = []
	count = 0
	for root, dir, files in os.walk(filename):
		for e in tf.train.summary_iterator(filename + '/' + files[0]):
			for v in e.summary.value:
				if v.tag == 'Train_AverageReturn':
					eval_returns.append(v.simple_value)
				elif v.tag == 'Train_BestReturn':
					best_returns.append(v.simple_value)
		break
	return eval_returns, best_returns

def plot_q1(logdir, head='../data/', scale=10000):
	eval_returns, best_returns = get_q1_data(head + logdir[0])
	x = np.arange(1, len(eval_returns)+1) * scale
	#plt.scatter(x, eval_returns)
	plt.plot(x, eval_returns, label='Train_AverageReturn')
	x = np.arange(1, len(best_returns)+1) * scale
	#plt.scatter(x, best_returns)
	plt.plot(x, best_returns, label='Train_BestReturn')
	plt.xlabel('Number of Iterations')
	plt.ylabel('Reward')
	plt.legend()
	plt.show()
	plt.savefig('../data/' + logdir[0] + '.png')
	plt.clf()

def plot_q2(dqn_logdir, ddqn_logdir, head='../data/', scale=10000):
	dqn_avg_eval_returns = np.mean([get_q1_data(head + log)[0] for log in dqn_logdir], axis=0)
	ddqn_avg_eval_returns = np.mean([get_q1_data(head + log)[0] for log in ddqn_logdir], axis=0)
	x = np.arange(1, len(dqn_avg_eval_returns)+1) * scale
	#plt.scatter(x, dqn_avg_eval_returns, label='DQN')
	plt.plot(x, dqn_avg_eval_returns, label='DQN')
	#plt.scatter(x, ddqn_avg_eval_returns, label='DDQN')
	plt.plot(x, ddqn_avg_eval_returns, label='DDQN')
	plt.xlabel('Number of Iterations')
	plt.ylabel('Train_AverageReturn')
	plt.legend()
	plt.show()
	plt.savefig('../data/' + '.png')
	plt.clf()


def main():
	q1_logdir = ['dqn_q1_PongNoFrameskip-v4_18-10-2019_14-51-46']
	plot_q1(q1_logdir)

	dqn_logdir = ['dqn_q2_dqn_1_LunarLander-v2_21-10-2019_11-16-11', 'dqn_q2_dqn_2_LunarLander-v2_21-10-2019_14-28-07', 'dqn_q2_dqn_3_LunarLander-v2_21-10-2019_15-08-58']
	ddqn_logdir = ['dqn_double_q_q2_doubledqn_1_LunarLander-v2_21-10-2019_00-34-25', 'dqn_double_q_q2_doubledqn_2_LunarLander-v2_21-10-2019_01-43-57', 'dqn_double_q_q2_doubledqn_3_LunarLander-v2_21-10-2019_02-28-06']
	plot_q2(dqn_logdir, ddqn_logdir)

	q3_head = '../data/'
	q3_logdir = ['dqn_q3_hparam1_PongNoFrameskip-v4_19-10-2019_07-24-22', 'dqn_q1_PongNoFrameskip-v4_18-10-2019_14-51-46', 'dqn_q3_hparam2_PongNoFrameskip-v4_19-10-2019_23-35-25', 'dqn_q3_hparam3_PongNoFrameskip-v4_21-10-2019_04-19-11']
	plot_data_same_graph(q3_logdir, 'q3', q3_head, metric='Train_AverageReturn', labels=['25000', '50000', '75000', '100000'], legend_title='learning_starts')

	q4_head = '../data/'
	q4_logdir = ['ac_1_1_CartPole-v0_16-10-2019_15-29-52', 'ac_100_1_CartPole-v0_16-10-2019_15-32-06', 'ac_1_100_CartPole-v0_16-10-2019_15-33-50', 'ac_10_10_CartPole-v0_16-10-2019_15-35-18']
	plot_data_same_graph(q4_logdir, 'q4', q4_head, scale=10)

	q5_logdir = ['ac_10_10_HalfCheetah-v2_16-10-2019_15-54-44', 'ac_10_10_InvertedPendulum-v2_16-10-2019_15-45-11']
	plot_data(q5_logdir, scale=1, title=['HalfCheetah', 'InvertedPendulum'])

if __name__ == '__main__':
	main()

