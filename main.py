import hiive.mdptoolbox
import hiive.mdptoolbox.mdp
import hiive.mdptoolbox.example
import numpy as np
import matplotlib.pyplot as plt

from gym.envs.toy_text.frozen_lake import generate_random_map
import hiive_openAI_extract

def performFrozenLakeExperiment():
    print("Perfroming Frozen Lake Experiment")

    openai_int = hiive_openAI_extract.OpenAI_MDPToolbox('FrozenLake-v0', True)
    states_small = openai_int.P
    rewards_small = openai_int.R
    random_map = generate_random_map(size=30, p=0.8)
    openai_int_big = hiive_openAI_extract.OpenAI_MDPToolbox('FrozenLake-v0', True, desc=random_map)
    states_large = openai_int_big.P
    rewards_large = openai_int_big.R

    print("Frozen Lake value iteration")

    iterations = 100;
    value_f_small = [0] * iterations
    iters_small = [0] * iterations
    time_array_small = [0] * iterations
    gamma_arr = [0] * iterations

    value_f_large = [0] * iterations
    iters_large = [0] * iterations
    time_array_large = [0] * iterations

    # Perform Value iteration for smaller size
    for i in range(0, iterations):
        vi_small = hiive.mdptoolbox.mdp.ValueIteration(states_small, rewards_small, (i + 0.5) / iterations, epsilon=0.1)
        vi_small.run()
        value_f_small[i] = np.mean(vi_small.V)
        iters_small[i] = vi_small.iter
        time_array_small[i] = vi_small.time

        # Perform Value iteration for larger size
        vi_large = hiive.mdptoolbox.mdp.ValueIteration(states_large, rewards_large, (i + 0.5) / iterations, epsilon=0.1)
        vi_large.run()
        value_f_large[i] = np.mean(vi_large.V)
        iters_large[i] = vi_large.iter
        time_array_large[i] = vi_large.time

        gamma_arr[i] = (i + 0.5) / iterations

    print("Value Iteration Less states")
    print(value_f_small)
    print(iters_small)
    print(time_array_small)

    print("Value Iteration Large states")
    print(value_f_large)
    print(iters_large)
    print(time_array_large)

    plt.plot(gamma_arr, iters_small, label=' 16 states')
    plt.plot(gamma_arr, iters_large, label=' 900 states')
    plt.xlabel('Gamma')
    plt.ylabel('Convergence')
    plt.title('MDP Frozen Lake - Value Iteration - Convergence plot')
    plt.grid()
    plt.legend()
    plt.savefig('Frozen_Lake_vi_convergence_iters')
    plt.clf()

    plt.plot(gamma_arr, time_array_small, label='16 states')
    plt.plot(gamma_arr, time_array_large, label='900 states')
    plt.xlabel('Gamma')
    plt.title('MDP Frozen Lake - Value Iteration - Execution Time plot')
    plt.ylabel('Execution Time')
    plt.grid()
    plt.legend()
    plt.savefig('Frozen_Lake_vi_time')
    plt.clf()

    plt.plot(gamma_arr, value_f_small, label='16 states')
    plt.plot(gamma_arr, value_f_large, label='900 states')
    plt.xlabel('Gamma')
    plt.ylabel('Mean Rewards')
    plt.title('MDP Frozen Lake - Value Iteration - Reward plot')
    plt.grid()
    plt.legend()
    plt.savefig('Frozen_Lake_vi_reward')
    plt.clf()


def performForestExperiment():
        ''' T - State Transition Matrix (S,a,S') i.e. probability of transitioning from state to state given an action is taken
        R - Reward matrix (S,a) i.e. rewards obtained in a state S when an action is taken a'''
        print("Performing Forest experiment")
        iterations = 100

        value_f_small = [0] * iterations
        policy_small = [0] * iterations
        iters_small = [0] * iterations
        time_array_small = [0] * iterations
        gamma_arr = [0] * iterations

        value_f_large = [0] * iterations
        policy_large = [0] * iterations
        iters_large = [0] * iterations
        time_array_large = [0] * iterations


        small_size = 10
        big_size = 1000

        T_Large, R_Large = hiive.mdptoolbox.example.forest(S=big_size)
        T_Small, R_Small = hiive.mdptoolbox.example.forest(S=small_size)

        # Perform Value iteration for smaller size
        for i in range(0, iterations):
            vi_small =  hiive.mdptoolbox.mdp.ValueIteration(T_Small, R_Small, (i+0.5)/iterations, epsilon=0.1)
            vi_small.run()
            value_f_small[i] = np.mean(vi_small.V)
            policy_small[i] = vi_small.policy
            iters_small[i] = vi_small.iter
            time_array_small[i] = vi_small.time

            # Perform Value iteration for larger size
            vi_large = hiive.mdptoolbox.mdp.ValueIteration(T_Large, R_Large, (i+0.5)/iterations, epsilon=0.1)
            vi_large.run()
            value_f_large[i] = np.mean(vi_large.V)
            policy_large[i] = vi_large.policy
            iters_large[i] = vi_large.iter
            time_array_large[i] = vi_large.time

            gamma_arr[i] = (i+0.5)/iterations

        print("Value Iteration Less states")
        print(value_f_small)
        print(iters_small)
        print(time_array_small)

        print("Value Iteration Large states")
        print(value_f_large)
        print(iters_large)
        print(time_array_large)

        plt.plot(gamma_arr, iters_small, label='10 states')
        plt.plot(gamma_arr, iters_large, label='1000 states')
        plt.xlabel('Gamma')
        plt.ylabel('Convergence')
        plt.title('MDP Forest - Value Iteration - Convergence plot')
        plt.grid()
        plt.legend()
        plt.savefig('Forest_vi_convergence_iters')
        plt.clf()

        plt.plot(gamma_arr, time_array_small, label='10 states')
        plt.plot(gamma_arr, time_array_large, label='1000 states')
        plt.xlabel('Gamma')
        plt.title('MDP Forest - Value Iteration - Execution Time plot')
        plt.ylabel('Execution Time')
        plt.grid()
        plt.legend()
        plt.savefig('Forest_vi_time')
        plt.clf()

        plt.plot(gamma_arr, value_f_small, label='10 states')
        plt.plot(gamma_arr, value_f_large, label='1000 states')
        plt.xlabel('Gamma')
        plt.ylabel('Mean Rewards')
        plt.title('MDP Forest - Value Iteration - Reward plot')
        plt.grid()
        plt.legend()
        plt.savefig('Forest_vi_reward')
        plt.clf()

        # Perform policy iteration
        value_f_small = [0] * iterations
        policy_small = [0] * iterations
        iters_small = [0] * iterations
        time_array_small = [0] * iterations
        gamma_arr = [0] * iterations

        value_f_large = [0] * iterations
        policy_large = [0] * iterations
        iters_large = [0] * iterations
        time_array_large = [0] * iterations


        for j in range(0, iterations):
            pi_small = hiive.mdptoolbox.mdp.PolicyIterationModified(T_Small, R_Small, (j+0.5)/iterations, epsilon=0.1)
            pi_small.run()
            value_f_small[j] = np.mean(pi_small.V)
            policy_small[j] = pi_small.policy
            iters_small[j] = pi_small.iter
            time_array_small[j] = pi_small.time

            pi_large = hiive.mdptoolbox.mdp.PolicyIterationModified(T_Large, R_Large, (j+0.5)/iterations, epsilon=0.1)
            pi_large.run()
            value_f_large[j] = np.mean(pi_large.V)
            policy_large[j] = pi_large.policy
            iters_large[j] = pi_large.iter
            time_array_large[j] = pi_large.time
            gamma_arr[j] = (j + 0.5) / iterations


        print("Policy_Iteration_small")
        print(value_f_small)
        print(iters_small)
        print(time_array_small)

        print("Policy_Iteration_large")
        print(value_f_large)
        print(iters_large)
        print(time_array_large)

        plt.plot(gamma_arr, time_array_small, label='10 states')
        plt.plot(gamma_arr, time_array_large, label='1000 states')
        plt.xlabel('Gamma')
        plt.title('MDP Forest - Policy Iteration - Execution Time plot')
        plt.ylabel('Execution Time')
        plt.grid()
        plt.legend()
        plt.savefig('Forest_pi_time')
        plt.clf()

        plt.plot(gamma_arr, value_f_small, label='10 states')
        plt.plot(gamma_arr, value_f_large, label='1000 states')
        plt.xlabel('Gamma')
        plt.ylabel('Mean Rewards')
        plt.title('MDP Forest - Policy Iteration - Reward plot')
        plt.grid()
        plt.legend()
        plt.savefig('Forest_pi_reward')
        plt.clf()

        plt.plot(gamma_arr, iters_small, label='10 states')
        plt.plot(gamma_arr, iters_large, label='1000 states')
        plt.xlabel('Gamma')
        plt.ylabel('Convergence')
        plt.title('MDP Forest - Policy Iteration - Convergence plot')
        plt.grid()
        plt.legend()
        plt.savefig('Forest_pi_convergence_iters')
        plt.clf()

if __name__ == '__main__':
    #performForestExperiment()
    performFrozenLakeExperiment()
