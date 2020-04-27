# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util
import numpy as np
from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        print("using discount {}".format(discount))
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        delta = 0.01
<<<<<<< HEAD

        for iteration in range(self.iterations):
            temp_values = util.Counter()
            l2_distance = 0
            for state in self.mdp.getStates():
                value = -np.inf
                if mdp.isTerminal(state):
                    temp_values[state] = 0
                    continue
                for action in mdp.getPossibleActions(state):
                    list = mdp.getTransitionStatesAndProbs(state, action)
                    tmp_value = 0
                    for pair in list:
                        tmp_value += pair[1] * (mdp.getReward(state, action, pair[0]) + self.discount * self.values[pair[0]])
                    value = max(value, tmp_value)
                temp_values[state] = value
                l2_distance = max(l2_distance, np.linalg.norm(value - self.values[state]))

            if l2_distance < delta:
                print(iteration)
                break
            self.values = temp_values

=======
>>>>>>> d00d5057d6ac8b04a3f737da09dbfae34a50aec3
        # TODO: Implement Value Iteration.
        # Exit either when the number of iterations is reached,
        # OR until convergence (L2 distance < delta).
        # Print the number of iterations to convergence.

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        # TODO: Implement this function according to the docs
        list = self.mdp.getTransitionStatesAndProbs(state, action)
        state = list[0][0]
        return self.values[state]

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        # TODO: Implement according to the doc
        if self.mdp.isTerminal(state):
            return None
        current_action = ""
        current_value = -np.inf

        for action in self.mdp.getPossibleActions(state):
            value = self.getQValue(state, action)
            if value > current_value:
                current_action = action
                current_value = value

        return current_action


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
