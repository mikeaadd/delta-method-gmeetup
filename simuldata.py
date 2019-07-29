import numpy as np
import pandas as pd
import abc
import ipdb


class SimulData(abc.ABC):
    name = NotImplemented  # type: str

    @abc.abstractmethod
    def data(self):
        pass

    @abc.abstractmethod
    def summary(self):
        pass

    @abc.abstractmethod
    def ground_truth(self):
        pass

    @abc.abstractmethod
    def update_mu(self):
        pass

    @abc.abstractmethod
    def get_increments(self):
        pass

class UncorBinom(SimulData):
    name = "Uncorrelated Binomial"

    def __init__(self, size, mu, n):
        self.size=size
        self.mu=mu
        self.n=n

    def summary(self):
        return f"mu:{self.mu} n:{self.n}"

    def data(self):
        session_ids=[np.repeat(i, val) for i, val in enumerate(np.random.poisson(3, self.n))]
        session_ids=[item for arr in session_ids for item in arr]
        Y=np.random.binomial(self.size, self.mu, len(session_ids))
        df=pd.DataFrame(data={'y': Y, 'session_id': session_ids})
        df_cluster=df.groupby('session_id', as_index=False)['y'].agg(['sum', 'count'])
        return {"unitlevel": df, "clusterlevel": df_cluster}

    def ground_truth(self):
        return self.size * self.mu

    def update_mu(self, mu):
        self.mu = mu

    def get_increments(self, delta=.1):
        top = 1 if self.mu + delta > 1 else self.mu + delta
        increments = np.linspace(0, top - self.mu, 30) + self.mu
        return increments


class HetBinom(SimulData):
    name = "Heterogeneous Binomial"

    def __init__(self, lamb, mu, sigma, n, probs):
        self.n=n
        self.lamb=lamb
        self.mu=mu
        self.sigma=sigma
        self.probs=np.array(probs)
    
    def summary(self):
        return f"n:{self.n} mu:{self.mu} lambda:{self.lamb} probs:{self.probs}"

    def data(self):
        nvec=np.random.multinomial(self.n, self.probs, 1)[0]
        K=nvec.sum()
        npage=np.random.poisson(np.repeat(self.lamb, nvec), K)
        pcr=np.maximum(0, np.minimum(1, np.random.normal(np.repeat(self.mu, nvec), np.repeat(self.sigma, nvec), K)))
        Y=np.random.binomial(1, np.repeat(pcr, npage), npage.sum())
        session_ids=np.repeat(list(range(1, K + 1)), npage)
        w=np.repeat(npage, npage)
        df=pd.DataFrame(data={'y': Y, 'session_id': session_ids, 'w': w})
        df_cluster=df.groupby('session_id', as_index=False)['y'].agg(['sum', 'count'])
        return {"unitlevel": df, "clusterlevel": df_cluster}

    def ground_truth(self):
        w=self.lamb * self.probs / (self.lamb * self.probs).sum()
        return (self.mu * w).sum()

    def update_mu(self, mu):
        self.mu[0] = mu

    def get_increments(self, delta=.3):
        top = 1 if self.mu[0] + delta > 1 else self.mu[0] + delta
        increments = np.linspace(0, top - self.mu[0], 30) + self.mu[0]
        return increments