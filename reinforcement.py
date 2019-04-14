import sys
import numpy as np
import math
import gym

def RT_init(problem, mutate=False):
    states_n = problem.observation_space.n
    actions_n = problem.action_space.n

    R = np.zeros((states_n, actions_n, states_n))
    T = np.zeros((states_n, actions_n, states_n))

    # Iterate over states, actions, and transitions
    for state in range(states_n):
        for action in range(actions_n):
            for transition in problem.env.P[state][action]:
                probability, next_state, reward, done = transition
                R[state, action, next_state] = reward
                T[state, action, next_state] = probability

            # Normalize T across state + action axes
            T[state, action, :] /= np.sum(T[state, action, :])

    return R, T

def value_iteration(problem, gamma=0.9, max_iterations=10**6, delta=10**-3):
    """ Runs Value Iteration on a gym problem """
    value = np.zeros(problem.observation_space.n)
    R, T = RT_init(problem)
        
    for i in range(max_iterations):
        prev_value = value.copy()
        Q = np.einsum('ijk,ijk -> ij', T, R + gamma * value)
        value = np.max(Q, axis=1)

        if np.max(np.abs(value - prev_value)) < delta:
            break

    # Get and return optimal policy
    policy = np.argmax(Q, axis=1)
    return policy, i + 1

def policy_iteration(problem, gamma=0.9, max_iterations=10**6, delta=10**-3):
    """ Runs Policy Iteration on a gym problem """
    states_n = problem.observation_space.n
    actions_n = problem.action_space.n

    # Initialize with a random policy and initial value function
    policy = np.array([problem.action_space.sample() for _ in range(states_n)])
    value = np.zeros(states_n)

    R, T = RT_init(problem)

    # Iterate and improve policies
    for i in range(max_iterations):
        prev_policy = policy.copy()

        for j in range(max_iterations):
            prev_value = value.copy()
            Q = np.einsum('ijk,ijk -> ij', T, R + gamma * value)
            value = np.sum(encode_policy(policy, (states_n, actions_n)) * Q, 1)

            if np.max(np.abs(prev_value - value)) < delta:
                break

        Q = np.einsum('ijk,ijk -> ij', T, R + gamma * value)
        policy = np.argmax(Q, axis=1)

        if np.array_equal(policy, prev_policy):
            break

    # Return optimal policy
    return policy, i + 1

if __name__ == "__main__":

    #mapping = {0: "L", 1: "D", 2: "R", 3: "U"}
    mdp = gym.make('FrozenLake8x8-v0')
    value_policy, value_iters = value_iteration(mdp)
    policy, iters = policy_iteration(mdp)

    diff = sum([abs(x-y) for x, y in zip(policy.flatten(), value_policy.flatten())])
    print("diff is")
    print(diff)
    print("policy is")
    print(policy)


    #mapping = {0: "S", 1: "N", 2: "E", 3: "W", 4: "P", 5: "D"}
    mdp = gym.make('Taxi-v2')
    value_policy, value_iters = value_iteration(mdp)
    policy, iters = policy_iteration(mdp)

    diff = sum([abs(x-y) for x, y in zip(policy.flatten(), value_policy.flatten())])
    print("diff is")
    print(diff)
    print("policy is")
    print(policy)

