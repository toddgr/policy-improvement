# Grace Todd
# Implementing Policy Improvement

import numpy as np

def pia(gamma, R, P):
    n = len(R) # Number of states
    states = list(range(n))

    m = len(P[0][0]) # Number of actions
    actions = list(range(m))

    # Policy iteration parameters
    iterations = 100  # termination criterion
    convergence_threshold = 0.000001 
    

    # Initialize pi
    # for every state s, look up the actions that can be taken there, 
    # and assign pi[s,a] = 1/(number of actions available at s)
    pi = [[0] * m for _ in range(n)]
    for s in states:    # n rows
            available_states = 0
            for a in actions:  # m actions  
                for s_prime in states:
                    if P[s][s_prime][a] != 0:   # As long as there is a probability of the action,
                        available_states += 1   # consider it an available state
            for _ in range(m):
                if available_states > 0:    # If there are any actions available for this state
                    pi[s][_] = 1 / available_states
                else:
                    pi[s][_] = 0


    for i in range(iterations):
        # Assume that the policy is stable, until a better policy comes along
        optimal_policy_found = True
        v = np.zeros(n) # Initialize v with zeros

        # Policy evaluation phase
        for j in range(iterations):
            maximum = 0  # Initialize max
            for s in range(n):
                val = 0 # current value function
                sum_pi_a = 0 # sum of all actions in pi
                sum_r_s_prime = 0  # sum of all s-prime state calculations
                for a in range(m):
                    for s_prime in states:
                    #s, r prime = sum of: prob[s][s'][a] * (rew[s][s'][a] + (gamma * v[s_prime]))
                        sum_r_s_prime += P[s][s_prime][a] * (R[s][s_prime][a] + (gamma * v[s_prime]))
                    sum_pi_a += pi[s][a]
                    
                    val = (sum_pi_a * sum_r_s_prime) # v(s) = sum a and sum s prime

                # Update maximum if value is no longer optimal
                maximum = max(maximum, abs(val - v[s]))

                v[s] = val  # Update with highest value
            # If maximum smaller than the convergence threshold for all states, current iteration terminates
            if maximum < convergence_threshold:
                break

        # Policy improvement
        # Check if policy needs to be optimized
        for s in states:
            val_max = v[s]
            val = 0

            for a in actions:
                for s_prime in states:
                    val += P[s][s_prime][a] * (R[s][s_prime][a] + (gamma * v[s_prime]))
                # Update policy if (i) action improves value and (ii) action different from current policy

                if val > val_max:
                    # Update policy by giving the state action certain probability
                    for _ in range(m):
                        pi[s][_] = 0
                    pi[s][a] = 1

                    val_max = val
                    optimal_policy_found = False

        # If policy didn't change, return pi
        if optimal_policy_found:
            break
    return pi