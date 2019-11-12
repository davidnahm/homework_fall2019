import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import re


def get_eval_avg_returns(filename, metric='Eval_AverageReturn'):
	eval_returns = []
	for root, dir, files in os.walk(filename):
		for e in tf.train.summary_iterator(filename + '/' + files[0]):
			for v in e.summary.value:
				if v.tag == metric:
					eval_returns.append(v.simple_value)
		break
	return eval_returns


def get_q2_data(filename):
	eval_returns = []
	train_returns = []
	count = 0
	for root, dir, files in os.walk(filename):
		print(filename)
		for e in tf.train.summary_iterator(filename + '/' + files[0]):
			print(files[0])
			for v in e.summary.value:
				if v.tag == 'Train_AverageReturn':
					train_returns.append(v.simple_value)
				elif v.tag == 'Eval_AverageReturn':
					eval_returns.append(v.simple_value)
		break
	return train_returns, eval_returns

def plot_q2(logdir, head='./cs285/data/'):
	event_acc = EventAccumulator(head + logdir[0])
	event_acc.Reload()
	_, step_nums, eval_vals = zip(*event_acc.Scalars('Eval_AverageReturn'))
	_, step_nums, train_vals = zip(*event_acc.Scalars('Train_AverageReturn'))
	x = np.arange(1, len(step_nums)+1)
	eval_returns = np.array(eval_vals)
	train_returns = np.array(train_vals)
	plt.scatter(x, eval_returns, label='Eval_AverageReturn')
	plt.scatter(x, train_returns, label='Train_AverageReturn')
	plt.xlabel('Number of Iterations')
	plt.ylabel('Reward')
	plt.title('Obstacles Single Iteration')
	plt.legend()
	plt.savefig('./cs285/data/figures/q2.png')
	plt.show()
	plt.clf()


def plot_q3(logdir, head='./cs285/data/', title=[], metric='Eval_AverageReturn'):
	data = []
	for i in range(len(logdir)):
		event_acc = EventAccumulator(head + logdir[i])
		event_acc.Reload()
		_, _, eval_vals = zip(*event_acc.Scalars('Eval_AverageReturn'))
		data.append(eval_vals)
	for i, dat in enumerate(data):
		x = np.arange(1, len(dat)+1)
		#plt.scatter(x, dat)
		plt.plot(x, dat)
		plt.xlabel('Number of Iterations')
		plt.ylabel(metric)
		plt.title(title[i])
		#plt.legend()
		plt.savefig('./cs285/data/figures/' + title[i] + '.png')
		plt.show()
		plt.clf()

def plot_q4(logdir, head='./cs285/data/', metric='Eval_AverageReturn'):
	titles = ['Effect of Ensemble Size', 'Effect of Number of Candidate Action Sequences', 'Effect of Planning Horizon']
	filenames = ['ensemble', 'numseq', 'horizon']
	for i in range(3):
		data = []
		for j in range(len(logdir[i])):
			event_acc = EventAccumulator(head + logdir[i][j])
			event_acc.Reload()
			_, _, eval_vals = zip(*event_acc.Scalars('Eval_AverageReturn'))
			data.append(eval_vals)
		for j, dat in enumerate(data):
			x = np.arange(1, len(dat)+1)
			#plt.scatter(x, dat)
			text = re.findall(r'[A-Za-z]+|\d+', logdir[i][j].split('_')[3])
			label = text[0] + '=' + text[1]
			plt.plot(x, dat, label=label)
		plt.xlabel('Number of Iterations')
		plt.ylabel(metric)
		plt.title(titles[i])
		plt.legend()
		plt.savefig('./cs285/data/figures/' + filenames[i] + '.png')
		plt.show()
		plt.clf()


def main():
	
	q2_logdir = ['mb_obstacles_singleiteration_obstacles-cs285-v0_31-10-2019_00-26-17']
	plot_q2(q2_logdir)

	q3_logdir = ['mb_obstacles_obstacles-cs285-v0_31-10-2019_00-34-57', 'mb_reacher_reacher-cs285-v0_31-10-2019_02-55-00', 'mb_cheetah_cheetah-cs285-v0_31-10-2019_10-47-26']
	plot_q3(q3_logdir, title=['Obstacles', 'Reacher', 'Cheetah'])

	q4_logdir = [['mb_q5_reacher_ensemble1_reacher-cs285-v0_02-11-2019_03-04-47', 
				 'mb_q5_reacher_ensemble3_reacher-cs285-v0_03-11-2019_01-50-58', 
				 'mb_q5_reacher_ensemble5_reacher-cs285-v0_04-11-2019_02-07-26'], 
				 ['mb_q5_reacher_numseq100_reacher-cs285-v0_01-11-2019_12-56-40',
				 'mb_q5_reacher_numseq1000_reacher-cs285-v0_01-11-2019_14-08-24'],
				 ['mb_q5_reacher_horizon5_reacher-cs285-v0_31-10-2019_22-49-11',
				 'mb_q5_reacher_horizon15_reacher-cs285-v0_01-11-2019_00-30-04',
				 'mb_q5_reacher_horizon30_reacher-cs285-v0_01-11-2019_07-28-11']]
	plot_q4(q4_logdir)


if __name__ == '__main__':
	main()

