"""
Run linear programming inverse reinforcement learning on the gridworld MDP.

Matthew Alger, 2015
matthew.alger@anu.edu.au
"""

import numpy as np
import matplotlib.pyplot as plt

import irl.linear_irl as linear_irl
import irl.mdp.gridworld as gridworld

def main(grid_size, discount, n_improvements = 5):
    """
    Run linear programming inverse reinforcement learning on the gridworld MDP.

    Plots the reward function.

    grid_size: Grid size. int.
    discount: MDP discount factor. float.
    """

    wind = 0.3
    trajectory_length = 3*grid_size

    gw = gridworld.Gridworld(grid_size, wind, discount)

    ground_r = np.array([gw.reward(s) for s in range(gw.n_states)])
    prob_optimal = 0.0
    rewards = []
    policies = []
    for i in range(n_improvements):
        policy = [gw.optimal_policy_improving(s,prob_optimal) for s in range(gw.n_states)]
        
        r = linear_irl.irl(gw.n_states, gw.n_actions, gw.transition_probability,
                policy, gw.discount, 1, 5)
        
        rewards.append(r)
        policies.append(policy)
#        print(r)
        
#        plt.subplot(1, 2, 1)
#        plt.pcolor(ground_r.reshape((grid_size, grid_size)))
#        plt.colorbar()
#        plt.title("Groundtruth reward")
#        plt.subplot(1, 2, 2)
#        plt.pcolor(r.reshape((grid_size, grid_size)))
#        plt.colorbar()
#        plt.title("Recovered reward")
#        plt.show()
        print(prob_optimal)
        prob_optimal += np.float(1/n_improvements)
    
    return rewards, policies

if __name__ == '__main__':
    n_improvements = 100
    rewards,policies = main(5, 0.2,n_improvements=n_improvements)
    x = np.arange(0.0,1.0,np.float(1/n_improvements))
    y = [r[24] for r in rewards]
