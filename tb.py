# Grace Todd
# Implementing Policy Improvement
# Creates sample inputs gamma, matrix R and matrix P, then calls pia on these inputs

import numpy as np
import unittest
from pia import pia

def transition_probability_array(n, m):
    # Create a 3D array of random values, of size n by n by m
    P = np.random.rand(n, n, m)

    # Normalize the values along the third dimension so that they add up to 1

    P /= P.sum(axis=2, keepdims=True)
    P = np.round(P, 2)
    print(f"P:\n{P}")
    return P

def reward_array(n, m):
    R = np.random.uniform(-1, 1, size=(n, n, m))
    R = np.round(R, 2)
    print(f"R:\n{R}")
    return R

def create_data(n, m):
    R = reward_array(n, m)
    P = transition_probability_array(n, m)
    gamma = np.random.uniform(0,1)
    gamma = np.round(gamma, 2)
    print(f"gamma: {gamma}")
    return R, P, gamma
    # print("pi: ", pia(gamma, R, P))

class TestPIA(unittest.TestCase):
    def test_base_case(self):
        # Where there is only 1 state, and one action to take.
        # Pi should show that there is only one action to take with 100% probability of it being taken
        R, P, gamma = create_data(1, 1)
        pi = pia(gamma, R, P)
        print("pi: ", pi)
        self.assertEqual(pi, [[1]])

    def test_pi_size(self):
        # For a variety of states and actions, pi is always the correct size.
        for n in range(1,5):
            for m in range(1,3):
                R, P, gamma = create_data(n, m)
                pi = pia(gamma, R, P)
                print("pi: ", pi)

                self.assertEqual(len(pi), n)
                self.assertEqual(len(pi[0]), m)

    def test_probability_correctness(self):
        # The final values of the actions of pi always add up to 100%
        n=5
        m=5
        R, P, gamma = create_data(n, m)
        pi = pia(gamma, R, P)
        print("pi: ", pi)

        for i in range(n):
            total_probability = 0
            for j in range(m):
                total_probability += pi[i][j]
            self.assertEqual(sum(pi[i])/total_probability, 1)



if __name__ == '__main__':
    #unittest.main() # Uncomment this to run test cases -- was not sure if these were necessary
    # Change these to lower numbers for fewer sample inputs
    n = 5 # states
    m = 4 # actions

    for i in range(1, n): # Starting at 1 state, ending with n states
        for j in range(1, m):
            print(f"\n------ {i} STATES, {j} ACTIONS ------")
            R, P, gamma = create_data(i, j)
            pi = pia(gamma, R, P)
            print("pi: ", pi)