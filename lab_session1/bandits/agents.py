"""
Module containing the agent classes to solve a Bandit problem.

Complete the code wherever TODO is written.
Do not forget the documentation for every class and method!
An example can be seen on the Bandit_Agent and Random_Agent classes.
"""
# -*- coding: utf-8 -*-
import numpy as np
import math
from utils import softmax, my_random_choice

class Bandit_Agent(object):
    """
    Abstract Agent to solve a Bandit problem.

    Contains the methods learn() and act() for the base life cycle of an agent.
    The reset() method reinitializes the agent.
    The minimum requirment to instantiate a child class of Bandit_Agent
    is that it implements the act() method (see Random_Agent).
    """
    def __init__(self, k:int, **kwargs):
        """
        Simply stores the number of arms of the Bandit problem.
        The __init__() method handles hyperparameters.
        Parameters
        ----------
        k: positive int
            Number of arms of the Bandit problem.
        kwargs: dictionary
            Additional parameters, ignored.
        """
        self.k = k

    def reset(self):
        """
        Reinitializes the agent to 0 knowledge, good as new.

        No inputs or outputs.
        The reset() method handles variables.
        """
        pass

    def learn(self, a:int, r:float):
        """
        Learning method. The agent learns that action a yielded reward r.
        Parameters
        ----------
        a: positive int < k
            Action that yielded the received reward r.
        r: float
            Reward for having performed action a.
        """
        pass

    def act(self) -> int:
        """
        Agent's method to select a lever (or Bandit) to pull.
        Returns
        -------
        a : positive int < k
            The action the agent chose to perform.
        """
        raise NotImplementedError("Calling method act() in Abstract class Bandit_Agent")

class Random_Agent(Bandit_Agent):
    """
    This agent doesn't learn, just acts purely randomly.
    Good baseline to compare to other agents.
    """
    def act(self):
        """
        Random action selection.
        Returns
        -------
        a : positive int < k
            A randomly selected action.
        """
        return np.random.randint(self.k)


class EpsGreedy_SampleAverage(Bandit_Agent):
    # This class uses Sample Averages to estimate q; others are non stationary.
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.averages = self.k*[0]
        self.eps = kwargs['eps']
        self.counts = self.k*[0]

    def reset(self):
        self.averages = self.k*[0]
        self.counts = self.k*[0]

    def learn(self, a:int, r:float):
        self.counts[a] += 1
        self.averages[a] += 1/self.counts[a]*(r - self.averages[a])

    def act(self) -> int:
        if np.random.uniform(0, 1) <= self.eps:
            return np.random.random_integers(0, self.k-1)
        else:
            max_score = max(self.averages)
            return self.averages.index(max_score)


class EpsGreedy(Bandit_Agent):
    # Non stationary agent with q estimating and eps-greedy action selection.
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.averages = self.k * [0]
        self.eps = kwargs['eps']
        self.alpha = kwargs['lr']
        self.counts = self.k * [0]

    def reset(self):
        self.averages = self.k * [0]
        self.counts = self.k * [0]

    def learn(self, a: int, r: float):
        self.counts[a] += 1
        self.averages[a] += self.alpha * (r - self.averages[a])

    def act(self) -> int:
        if np.random.uniform(0, 1) <= self.eps:
            return np.random.random_integers(0, self.k - 1)
        else:
            max_score = max(self.averages)
            return self.averages.index(max_score)


class OptimisticGreedy(EpsGreedy_SampleAverage):
    # Same as above but with optimistic starting values.
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.q0 = kwargs['q0']
        self.averages = self.k * [self.q0]  # set averages to the optimistic value
        self.eps = 0

    def reset(self):
        self.averages = self.k * [self.q0]

class UCB(Bandit_Agent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.counts = self.k * [0]
        self.averages = self.k * [0]
        self.alpha = kwargs['alpha']
        self.c = kwargs['c']
        self.time = 1

    def reset(self):
        self.counts = self.k * [0]
        self.averages = self.k * [0]
        self.time = 1

    def learn(self, a:int, r:float):
        self.counts[a] += 1
        self.averages[a] += self.alpha * (r - self.averages[a])

    def act(self) -> int:
        values = [self.averages[a] + self.c*math.sqrt(math.log(self.time)/self.counts[a]) if (self.counts[a] != 0) else np.inf for a in range(self.k)]
        return values.index(max(values))



class Gradient_Bandit(Bandit_Agent):
    # TODO: Fix

    # If you want this to run fast, use the my_random_choice function from
    # utils instead of np.random.choice to sample from the softmax
    # You can also find the softmax function in utils.
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.preferences = self.k * [0]
        self.sum_reward = 0
        self.count = 0
        self.alpha = kwargs['alpha']
        self.probas = self.k*[1/self.k]

    def reset(self):
        self.sum_reward = 0
        self.count = 0
        self.probas = self.k * [1/self.k]
        self.preferences = self.k * [0]

    def learn(self, a:int, r:float):
        self.count += 1
        self.sum_reward += r
        avg_reward = self.sum_reward/self.count
        d_reward = (r-avg_reward)
        self.preferences = [self.preferences[i] - self.alpha*d_reward*self.probas[i] if (i != a)
                            else self.preferences[i] + self.alpha*d_reward*(1-self.probas[i])
                            for i in range(self.k)]

    def act(self) -> int:
        self.probas = softmax(self.preferences)
        return my_random_choice(self.k, self.probas)
