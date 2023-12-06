# Implementing the Policy Improvement Algorithm

This project aims to find the optimal policy (or, the most efficient sequence of states and actions for an agent) for Markov's Decision Process models in Python. The actual policy improvement algorithm is in the `pia.py` file, and benchmark test cases can be found and applied if `tb.py` is run. The test cases are not as ideal as I would like them to be... in future commits to this repo, I hope to improve them a bit and make them a little less conditional/random.

## Policy Improvement
Policy improvement can be broken down into three steps: initialization, policy evaluation, and policy improvement. In other words-- we start with a policy, we check if it needs to be improved, and if it does, then we improve it. This algorithm should theoretically return the most optimal policy, or the most optimal series of states and actions that an agent should take. 

## Inputs and Outputs
For this program, the benchmark test cases provide the policy improvement algorithm (pia) with a probability matrix `P` (i.e. how likely is it that a certain action will be taken, given the current state), a reward matrix `R` (i.e. what is the reward for taking an action given the current state), and a discount factor `gamma`, which ensures that there isn't an infinite amount of reward given.

My goal with this project is to better understand one method of choosing an optimal policy, and how this impacts machine learning practices and optimal policy outputs.
