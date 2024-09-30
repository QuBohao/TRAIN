# --coding:utf-8--

import time
import datetime

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


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# torch.autograd.set_detect_anomaly(True)


class args(object):
    env_name = 'halfcheetah-medium-expert-v2'
    # env_name = 'walker2d-medium-expert-v2'
    # env_name = 'hopper-medium-expert-v2'
    # env_name = 'hammer-human-v0'
    # env_name = 'door-human-v0'
    # env_name = 'relocate-human-v0'

    seed = 1234
    batch_size = 50000
    random_mask_ratio = 0.8
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    nn_iteration = 5  
    p_iteration = 10  
    plot = False


class MLPNetwork(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_size=128):
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
    print(env_name)
    env = gym.make(env_name)

    # observations: An N by observation dimensional array of observations.
    # actions: An N by action dimensional array of actions.
    # rewards: An N dimensional array of rewards.
    # terminals: An N dimensional array of episode termination flags. This is true when episodes end due to termination conditions such as falling over.
    # timeouts: An N dimensional array of termination flags. This is true when episodes end due to reaching the maximum episode length.
    # infos: Contains optional task-specific debugging information.

    dataset = d4rl.qlearning_dataset(env)

    return dataset


# ... (previous imports and class definitions)

def get_weight_matrix(total_num, observations, actions, network, args):
    #obs_tensor = torch.from_numpy(observations).to(args.device)
    #act_tensor = torch.from_numpy(actions).to(args.device)

    #obs_dist = torch.cdist(obs_tensor, obs_tensor)
    #act_dist = torch.cdist(act_tensor, act_tensor)

    obs_tensor = observations
    act_tensor = actions

    obs_act_tensor = torch.cat((obs_tensor, act_tensor), dim=1)

    dist = torch.cdist(obs_act_tensor,obs_act_tensor)

    dist_matrix = torch.exp(-network(dist))

    # Compute sums and normalize
    sums = dist_matrix.sum(dim=1, keepdim=True)
    p_matrix = dist_matrix / sums

    return p_matrix

def get_final_weight_matrix(total_num, observations, actions, network, args):
    #obs_tensor = torch.from_numpy(observations).to(args.device)
    #act_tensor = torch.from_numpy(actions).to(args.device)

    #obs_dist = torch.cdist(obs_tensor, obs_tensor)
    #act_dist = torch.cdist(act_tensor, act_tensor)

    weight_matrix = torch.zeros(total_num, total_num).to(args.device)

    submatrix_size = int(math.ceil(args.batch_size * (1-args.random_mask_ratio)))

    obs_tensor = observations
    act_tensor = actions

    obs_act_tensor = torch.cat((obs_tensor, act_tensor), dim=1)

    for i in range(5):
        for j in range(5):
            temp = obs_act_tensor[i*submatrix_size:(i+1)*submatrix_size, j*submatrix_size:(j+1)*submatrix_size]

            dist = torch.cdist(temp,temp)

            weight_matrix[i*submatrix_size:(i+1)*submatrix_size, j*submatrix_size:(j+1)*submatrix_size] = torch.exp(-network(dist))

    # Compute sums and normalize
    sums = weight_matrix.sum(dim=1, keepdim=True)
    p_matrix = weight_matrix / sums

    return p_matrix

def propagate_labels(weight_matrix, label_set, rewarded_index, labelled_rewards, args):
    propagated_labelmatrix = weight_matrix.mm(label_set)
    for _ in tqdm(range(args.p_iteration), desc='Reward Propagation'):
        propagated_labelmatrix = weight_matrix.mm(propagated_labelmatrix)
        #propagated_labelmatrix[rewarded_index] = propagated_labelmatrix[rewarded_index].t().fill_(labelled_rewards)
        # clamp label
        propagated_labelmatrix[rewarded_index] = torch.transpose(labelled_rewards, 0, 1)

    return propagated_labelmatrix

def construct_and_train_graph(observations, actions, rewards, args):

    input_dim = int(math.ceil(args.batch_size * (1-args.random_mask_ratio)))

    theta = MLPNetwork(input_dim=input_dim, output_dim=1, hidden_size=64).to(args.device)
    optimizer = opt.Adam(theta.parameters(), lr=3e-4)

    # Randomly mask a portion of rewards
    total_num = rewards.size
    mask_num = int(total_num * args.random_mask_ratio)
    mask_ind = np.random.choice(total_num, mask_num, replace=False)
    rewards[mask_ind] = 0

    label_set = torch.from_numpy(rewards).view(-1, 1).to(args.device)

    # Remove masked elements to get the labeled dataset
    labelled_mask = np.delete(np.arange(total_num), mask_ind)
    labelled_observations = torch.from_numpy(observations[labelled_mask]).to(args.device)
    labelled_actions = torch.from_numpy(actions[labelled_mask]).to(args.device)
    labelled_rewards_tensor = torch.from_numpy(rewards[labelled_mask]).view(1, -1).to(args.device)

    for _ in tqdm(range(args.nn_iteration), desc='Network training'):
        labelled_p_matrix = get_weight_matrix(labelled_observations.size(0), labelled_observations, labelled_actions, theta, args)

        predicted_rewards = labelled_rewards_tensor.mm(labelled_p_matrix)

        criterion = nn.MSELoss().to(args.device)
        loss = criterion(predicted_rewards, labelled_rewards_tensor)

        print("Loss: {}".format(loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("====== Graph training finished ======")

    observations = torch.from_numpy(observations).to(args.device)
    actions = torch.from_numpy(actions).to(args.device)

    # Calculate weight matrix for the entire dataset using the trained theta
    p_matrix = get_final_weight_matrix(total_num, observations, actions, theta, args)


    # Propagate labels using the weight matrix
    propagated_labelmatrix = propagate_labels(p_matrix, label_set, labelled_mask, labelled_rewards_tensor, args)

    print("====== Reward propagation finished ======")

    return propagated_labelmatrix

# ... (rest of the code remains unchanged)


if __name__ == '__main__':

    start=time.clock()
    # start=datetime.datetime.now()
    print("start_time:", start)

    args = args()
    # args.env_name = 'halfcheetah-medium-expert-v2'
    dataset = get_dataset(args)

    dataset_observations_size = dataset['observations'].shape[0]
    dataset_actions_size = dataset['actions'].shape[0]
    assert dataset_observations_size == dataset_actions_size, "Error, dataset observations_size is not equal to actions_size."

    # for i_epoch in range(int(dataset['observations'].size / args.batch_size)):
    #     print(i_epoch)


    observations = dataset['observations']
    actions = dataset['actions']
    rewards = dataset['rewards']

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


    np.savetxt('./data/batchsize_{}_ratio_{}_nn_{}_p_{}_mse_loss_{}.csv'.format(args.batch_size, args.random_mask_ratio,
                                                                         args.nn_iteration, args.p_iteration, mse_loss),
               np.column_stack((x, y1, y2)), delimiter=',')

    print("save data succefull_batchsize_{}_ratio_{}_nn_{}_p_{}_mse_loss_{}".format(args.batch_size, args.random_mask_ratio,
                                                                             args.nn_iteration, args.p_iteration, mse_loss))

    end = time.clock()
    # end = datetime.datetime.now()
    print("end_time:", end)

    # run_time = end-start
    run_time = end - start
    print("run_time:", run_time)


    if args.plot:
 
        plt.plot(x, y1, label='propagated')
        plt.plot(x, y2, label='ground_truth')


        plt.title('Reward Compare')
        plt.xlabel('x')
        plt.ylabel('y')


        plt.legend()


        plt.show()



  