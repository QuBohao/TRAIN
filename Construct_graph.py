# --coding:utf-8--

import d4rl  # Import required to register environments
import gym
import numpy as np
from numpy import *
from scipy.spatial import distance

from Utils import *
from tqdm import *
import torch.nn as nn
import torch.optim as opt
import torch
import torch.nn.functional

import matplotlib.pyplot as plt
import os

# 设置 CUDA_VISIBLE_DEVICES 环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# torch.autograd.set_detect_anomaly(True)


class args(object):
    env_name = 'halfcheetah-medium-expert-v2'
    seed = 1234
    batch_size = 2000
    random_mask_ratio = 0.8
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    nn_iteration = 20  # 神经网络迭代次数
    p_iteration = 10  # 传播次数
    plot = False
    norm = 2.5


class MLPNetwork(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_size=256):
        super(MLPNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            #     nn.Linear(hidden_size, hidden_size),
            #     nn.ReLU(),
            #     nn.Linear(hidden_size, hidden_size),
            #     nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
        )
        # self.network = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.network(x)


def get_dataset(args):
    env_name = args.env_name
    env = gym.make(env_name)

    # observations: An N by observation dimensional array of observations.
    # actions: An N by action dimensional array of actions.
    # rewards: An N dimensional array of rewards.
    # terminals: An N dimensional array of episode termination flags. This is true when episodes end due to termination conditions such as falling over.
    # timeouts: An N dimensional array of termination flags. This is true when episodes end due to reaching the maximum episode length.
    # infos: Contains optional task-specific debugging information.

    dataset = d4rl.qlearning_dataset(env)

    return dataset


def get_weight_matrix(total_num, observations, actions, network, args):
    weight_matrix = torch.zeros(total_num, total_num).to(args.device)
    # p_matrix = torch.zeros(total_num, total_num).to(args.device)

    p_norm = args.norm
    # get weight matrix
    for i in range(total_num):
        start = i
        for j in range(start, total_num):
            if i == j:
                weight_matrix[j][i] = 0
            else:
                weight_matrix[j][i] = weight_matrix[i][j] = torch.exp(-(network(torch.tensor(
                    [torch.dist(observations[i][0], observations[j][0], p=p_norm),
                     torch.dist(observations[i][1], observations[j][1], p=p_norm),
                     torch.dist(observations[i][2], observations[j][2], p=p_norm),
                     torch.dist(observations[i][3], observations[j][3], p=p_norm),
                     torch.dist(observations[i][4], observations[j][4], p=p_norm),
                     torch.dist(observations[i][5], observations[j][5], p=p_norm),
                     torch.dist(observations[i][6], observations[j][6], p=p_norm),
                     torch.dist(observations[i][7], observations[j][7], p=p_norm),
                     torch.dist(observations[i][8], observations[j][8], p=p_norm),
                     torch.dist(observations[i][9], observations[j][9], p=p_norm),
                     torch.dist(observations[i][10], observations[j][10], p=p_norm),
                     torch.dist(observations[i][11], observations[j][11], p=p_norm),
                     torch.dist(observations[i][12], observations[j][12], p=p_norm),
                     torch.dist(observations[i][13], observations[j][13], p=p_norm),
                     torch.dist(observations[i][14], observations[j][14], p=p_norm),
                     torch.dist(observations[i][15], observations[j][15], p=p_norm),
                     torch.dist(observations[i][16], observations[j][16], p=p_norm),
                     torch.dist(actions[i][0], actions[j][0], p=p_norm),
                     torch.dist(actions[i][1], actions[j][1], p=p_norm),
                     torch.dist(actions[i][2], actions[j][2], p=p_norm),
                     torch.dist(actions[i][3], actions[j][3], p=p_norm),
                     torch.dist(actions[i][4], actions[j][4], p=p_norm),
                     torch.dist(actions[i][5], actions[j][5], p=p_norm)
                     ]).to(args.device))))

    # 计算每行的和
    sums = torch.sum(weight_matrix, dim=1, keepdim=True)

    # 将每个元素除以对应的和
    p_matrix = torch.div(weight_matrix, sums)

    return p_matrix


# Iterating the dot product of randomwalk matrix with the filled Y0 matrix to propagate labels
def propagate_labels(weight_matrix, label_set, rewarded_index, labelled_rewards, args):
    propagated_labelmatrix = torch.matmul(weight_matrix, label_set)
    for k in tqdm(range(args.p_iteration), desc='Reward Propagation'):
        propagated_labelmatrix = torch.matmul(weight_matrix, propagated_labelmatrix)
        # clamp label
        propagated_labelmatrix[rewarded_index] = torch.transpose(labelled_rewards, 0, 1)

    return propagated_labelmatrix


def construct_and_train_graph(observations, actions, rewards, args):
    theta = MLPNetwork(input_dim=23, output_dim=1, hidden_size=23).to(args.device)
    optimizer = opt.Adam(theta.parameters(), lr=3e-4)

    # 根据比例将数据集中的reward抹掉
    total_num = rewards.size
    mask_num = int(total_num * args.random_mask_ratio)
    mask_ind = np.random.choice(total_num, mask_num, replace=False)
    rewards[mask_ind] = 0

    label_set = torch.unsqueeze(torch.from_numpy(rewards), 1).to(
        args.device)  # label set containing known and unknown labels

    # 根据索引删除元素,获得有标签的数据集
    labelled_observations = np.delete(observations, [mask_ind], axis=0)
    labelled_actions = np.delete(actions, [mask_ind], axis=0)
    labelled_rewards = np.delete(rewards, [mask_ind])
    labelled_total_num = labelled_observations.shape[0]

    # 获取有reward的index
    all_index = np.arange(rewards.size)
    rewarded_index = np.delete(all_index, [mask_ind])

    labelled_observations = torch.from_numpy(labelled_observations).to(args.device)
    labelled_actions = torch.from_numpy(labelled_actions).to(args.device)
    labelled_rewards_tensor = torch.unsqueeze(torch.from_numpy(labelled_rewards), 0).to(args.device)

    for i in tqdm(range(args.nn_iteration), desc='Network training'):
        labelled_p_matrix = get_weight_matrix(labelled_total_num, labelled_observations, labelled_actions, theta, args)

        predicted_rewards = torch.matmul(labelled_rewards_tensor, labelled_p_matrix)

        criterion = nn.MSELoss().cuda()
        # criterion = nn.L1Loss().cuda()
        loss = criterion(predicted_rewards, labelled_rewards_tensor)

        print("Iteration:{},loss:{}".format(i, loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("====== graph training finish ======")

    observations = torch.from_numpy(observations).to(args.device)
    actions = torch.from_numpy(actions).to(args.device)

    # 用训练好的theta计算所有数据的weight_matrix
    p_matrix = get_weight_matrix(total_num, observations, actions, theta, args)

    # 用计算好的weight_matrix和已有的labelled_rewards_tensor推算其他标签.
    propagated_labelmatrix = propagate_labels(p_matrix, label_set, rewarded_index, labelled_rewards_tensor, args)

    print("====== reward propagation finish ======")

    return propagated_labelmatrix


if __name__ == '__main__':
    args = args()
    args.env_name = 'halfcheetah-medium-expert-v2'
    dataset = get_dataset(args)

    dataset_observations_size = dataset['observations'].shape[0]
    dataset_actions_size = dataset['actions'].shape[0]
    assert dataset_observations_size == dataset_actions_size, "Error, dataset observations_size is not equal to actions_size."

    # for i_epoch in range(int(dataset['observations'].size / args.batch_size)):
    #     print(i_epoch)

    # 全部数据集
    observations = dataset['observations']
    actions = dataset['actions']
    rewards = dataset['rewards']

    # 根据比例 获取一个batch的数据
    total_num = dataset['rewards'].shape[0]

    # batch_ind = np.random.choice(total_num, args.batch_size, replace=False)

    random_index = np.random.randint(low=0, high=(total_num - args.batch_size), size=1, dtype='int')[0]
    batch_ind = np.arange(start=random_index, stop=(random_index+args.batch_size))

    batch_observations = observations[batch_ind]
    batch_actions = actions[batch_ind]
    batch_rewards = rewards[batch_ind]

    propagated_labelmatrix = construct_and_train_graph(batch_observations, batch_actions, batch_rewards, args)

    y1 = propagated_labelmatrix.cpu().detach().numpy()

    y2 = rewards[batch_ind]  # ground_truth

    x = np.arange(y1.size)

    mse_loss = np.mean((y2 - y1) ** 2)

    print("MSE loss:{}".format(mse_loss))

    # 将数据保存到 CSV 文件中
    np.savetxt('./data/batchsize_{}_ratio_{}_nn_{}_p_{}_mse_loss_{}_norm_{}.csv'.format(args.batch_size, args.random_mask_ratio,
                                                                         args.nn_iteration, args.p_iteration, mse_loss, args.norm),
               np.column_stack((x, y1, y2)), delimiter=',')

    print("save data succefull_batchsize_{}_ratio_{}_nn_{}_p_{}_mse_loss_{}".format(args.batch_size, args.random_mask_ratio,
                                                                             args.nn_iteration, args.p_iteration, mse_loss))
    if args.plot:
        # 绘制两条曲线
        plt.plot(x, y1, label='propagated')
        plt.plot(x, y2, label='ground_truth')

        # 添加标题和标签
        plt.title('Reward Compare')
        plt.xlabel('x')
        plt.ylabel('y')

        # 显示图例
        plt.legend()

        # 显示图像
        plt.show()

    print("===============over====================")

    