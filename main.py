import time

import gym
import hiive.mdptoolbox
import hiive.mdptoolbox.mdp
import hiive.mdptoolbox.example
import numpy as np
import matplotlib.pyplot as plt

from gym.envs.toy_text.frozen_lake import generate_random_map
import hiive_openAI_extract

def performFrozenLakeExperiment():
    print("Perfroming Frozen Lake Experiment")

    '''openai_int = hiive_openAI_extract.OpenAI_MDPToolbox('FrozenLake-v0', True)
    states_small = openai_int.P
    rewards_small = openai_int.R
    random_map = generate_random_map(size=20, p=0.8)
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

    plt.plot(gamma_arr, iters_small, label='16 states')
    plt.plot(gamma_arr, iters_large, label='400 states')
    plt.xlabel('Gamma')
    plt.ylabel('Convergence')
    plt.title('MDP Frozen Lake - Value Iteration - Convergence plot')
    plt.grid()
    plt.legend()
    plt.savefig('Frozen_Lake_vi_convergence_iters')
    plt.clf()

    plt.plot(gamma_arr, time_array_small, label='16 states')
    plt.plot(gamma_arr, time_array_large, label='400 states')
    plt.xlabel('Gamma')
    plt.title('MDP Frozen Lake - Value Iteration - Execution Time plot')
    plt.ylabel('Execution Time')
    plt.grid()
    plt.legend()
    plt.savefig('Frozen_Lake_vi_time')
    plt.clf()

    plt.plot(gamma_arr, value_f_small, label='16 states')
    plt.plot(gamma_arr, value_f_large, label='400 states')
    plt.xlabel('Gamma')
    plt.ylabel('Mean Rewards')
    plt.title('MDP Frozen Lake - Value Iteration - Reward plot')
    plt.grid()
    plt.legend()
    plt.savefig('Frozen_Lake_vi_reward')
    plt.clf()

    value_f_small = [0] * iterations
    iters_small = [0] * iterations
    time_array_small = [0] * iterations
    gamma_arr = [0] * iterations

    value_f_large = [0] * iterations
    iters_large = [0] * iterations
    time_array_large = [0] * iterations

    # Perform Policy iteration for smaller size
    for i in range(0, iterations):
        pi_small = hiive.mdptoolbox.mdp.PolicyIterationModified(states_small, rewards_small, (i + 0.5) / iterations, epsilon=0.1)
        pi_small.run()
        value_f_small[i] = np.mean(pi_small.V)
        iters_small[i] = pi_small.iter
        time_array_small[i] = pi_small.time

        # Perform Value iteration for larger size
        pi_large = hiive.mdptoolbox.mdp.PolicyIterationModified(states_large, rewards_large, (i + 0.5) / iterations, epsilon=0.1)
        pi_large.run()
        value_f_large[i] = np.mean(pi_large.V)
        iters_large[i] = pi_large.iter
        time_array_large[i] = pi_large.time

        gamma_arr[i] = (i + 0.5) / iterations

    print("Policy Iteration Less states")
    print(value_f_small)
    print(iters_small)
    print(time_array_small)

    print("Policy Iteration Large states")
    print(value_f_large)
    print(iters_large)
    print(time_array_large)

    plt.plot(gamma_arr, iters_small, label='16 states')
    plt.plot(gamma_arr, iters_large, label='400 states')
    plt.xlabel('Gamma')
    plt.ylabel('Convergence')
    plt.title('MDP Frozen Lake - Policy Iteration - Convergence plot')
    plt.grid()
    plt.legend()
    plt.savefig('Frozen_Lake_pi_convergence_iters')
    plt.clf()

    plt.plot(gamma_arr, time_array_small, label='16 states')
    plt.plot(gamma_arr, time_array_large, label='400 states')
    plt.xlabel('Gamma')
    plt.title('MDP Frozen Lake - Policy Iteration - Execution Time plot')
    plt.ylabel('Execution Time')
    plt.grid()
    plt.legend()
    plt.savefig('Frozen_Lake_pi_time')
    plt.clf()

    plt.plot(gamma_arr, value_f_small, label='16 states')
    plt.plot(gamma_arr, value_f_large, label='400 states')
    plt.xlabel('Gamma')
    plt.ylabel('Mean Rewards')
    plt.title('MDP Frozen Lake - Policy Iteration - Reward plot')
    plt.grid()
    plt.legend()
    plt.savefig('Frozen_Lake_pi_reward')
    plt.clf()'''

    # Q learning for frozen lake
    print('Q LEARNING WITH FROZEN LAKE')
    environment = 'FrozenLake-v0'
    env = gym.make(environment)
    env = env.unwrapped
    desc = env.unwrapped.desc

    st = time.time()
    reward_array = []
    iter_array = []
    size_array = []
    chunks_array = []
    averages_array = []
    time_array = []
    Q_array = []
    epsilon_values = [0.05, 0.15, 0.25, 0.5, 0.75, 0.95]

    st = time.time()
    for epsilon in [0.05, 0.15, 0.25, 0.5, 0.75, 0.90]:
        Q = np.zeros((env.observation_space.n, env.action_space.n))
        rewards = []
        iters = []
        optimal = [0] * env.observation_space.n
        alpha = 0.85
        gamma = 0.95
        episodes = 30000
        environment = 'FrozenLake-v0'
        env = gym.make(environment)
        env = env.unwrapped
        desc = env.unwrapped.desc
        for episode in range(episodes):
            state = env.reset()
            done = False
            t_reward = 0
            max_steps = 1000000
            for i in range(max_steps):
                if done:
                    break
                current = state
                if np.random.rand() < (epsilon):
                    action = np.argmax(Q[current, :])
                else:
                    action = env.action_space.sample()

                state, reward, done, info = env.step(action)
                t_reward += reward
                Q[current, action] += alpha * (reward + gamma * np.max(Q[state, :]) - Q[current, action])
            epsilon = (1 - 2.71 ** (-episode / 1000))
            rewards.append(t_reward)
            iters.append(i)

        for k in range(env.observation_space.n):
            optimal[k] = np.argmax(Q[k, :])

        reward_array.append(rewards)
        iter_array.append(iters)
        Q_array.append(Q)

        env.close()
        end = time.time()
        # print("time :",end-st)
        time_array.append(end - st)

        # Plot results
        def chunk_list(l, n):
            for i in range(0, len(l), n):
                yield l[i:i + n]

        size = int(episodes / 50)
        chunks = list(chunk_list(rewards, size))
        averages = [sum(chunk) / len(chunk) for chunk in chunks]
        size_array.append(size)
        chunks_array.append(chunks)
        averages_array.append(averages)

        print(size_array)
        print(chunks_array)
        print(averages_array)



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

        # Perform Q learning
        print('Q LEARNING WITH FOREST MANAGEMENT')
        time_array_small = []
        reward_small = []
        time_array_large = []
        reward_large = []
        alpha_values = [0.05, 0.15, 0.25, 0.5, 0.75, 0.95]
        np.random.seed(100)
        for alpha in alpha_values:
            ql = hiive.mdptoolbox.mdp.QLearning(T_Small, R_Small, 0.95,alpha_min=alpha)
            st = time.time()
            ql.run()
            end = time.time()
            time_array_small.append(end - st)
            reward_small.append(np.max(ql.V))

            ql_large = hiive.mdptoolbox.mdp.QLearning(T_Large, R_Large, 0.95, alpha_min=alpha)
            st_large = time.time()
            ql_large.run()
            end_large = time.time()
            time_array_large.append(end_large - st_large)
            reward_large.append(np.max(ql_large.V))


        print(alpha_values)

        print(time_array_small)
        print(reward_small)

        print(time_array_large)
        print(reward_large)

if __name__ == '__main__':
    performForestExperiment()
    #performFrozenLakeExperiment()
