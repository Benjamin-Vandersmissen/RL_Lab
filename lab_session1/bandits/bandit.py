"""
Module containing the k-armed bandit problem
Complete the code wherever TODO is written.
Do not forget the documentation for every class and method!
We expect all classes to follow the Bandit abstract object formalism.
"""
# -*- coding: utf-8 -*-
import numpy as np

class Bandit(object):
    """
    Abstract concept of a Bandit, i.e. Slot Machine, the Agent can pull.

    A Bandit is a distribution over reals.
    The pull() method samples from the distribution to give out a reward.
    """
    def __init__(self, **kwargs):
        """
        Empty for our simple one-armed bandits, without hyperparameters.
        Parameters
        ----------
        **kwargs: dictionary
            Ignored additional inputs.
        """
        pass

    def reset(self):
        """
        Reinitializes the distribution.
        """
        pass

    def pull(self) -> float:
        """
        Returns a sample from the distribution.
        """
        raise NotImplementedError("Calling method pull() in Abstract class Bandit")


class Gaussian_Bandit(Bandit):
    # Reminder: the Gaussian_Bandit's distribution is a fixed Gaussian.
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.m = np.random.normal(0, 1)

    def reset(self):
        self.m = np.random.normal(0, 1)

    def pull(self) -> float:
        return np.random.normal(self.m, 1)

class Gaussian_Bandit_NonStat(Gaussian_Bandit):
    # Reminder: the distribution mean changes each step over time,
    # with increments following N(m=0,std=0.01)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orig_m = self.m

    def reset(self):
        super().reset()
        self.m = self.orig_m

    def update(self):
        self.m += np.random.normal(0, 0.01)

    def pull(self) -> float:
        return np.random.normal(self.m, 1)

class KBandit:
    # Reminder: The k-armed Bandit is a set of k Bandits.
    # In this case we mean for it to be a set of Gaussian_Bandits.

    def __init__(self, **kwargs):
        self.bandits = [Gaussian_Bandit() for i in range(kwargs['k'])]

    def reset(self):
        for bandit in self.bandits:
            bandit.reset()

    def pull(self, i) -> float:
        return self.bandits[i].pull()


class KBandit_NonStat(KBandit):
    # Reminder: Same as KBandit, with non stationary Bandits.

    def __init__(self, **kwargs):
        self.bandits = [Gaussian_Bandit_NonStat() for i in range(kwargs['k'])]

    def pull(self, i) -> float:
        r = super().pull(i)
        for bandit in self.bandits : bandit.update()
        return r

    def reset(self):
        for bandit in self. bandits:
            bandit.reset()
