import numpy as np
import pandas as pd
import abc
import copy
from scipy import stats
import math
import matplotlib.pyplot as plt
from itertools import chain
from abmethods import Naive, MixedModel, GroupNoSpark, DeltaMethod
from simuldata import HetBinom, UncorBinom
import ipdb

class Simulation:
    def __init__(self, m, simuldata, tests):
        self.m = m
        self.simuldata = simuldata
        self.tests = tests

    def run(self):
        res = np.zeros([len(self.tests), 2, self.m])
        for i in range(self.m):
            res[:, :, i] = self.run_once()
            if i % 20 == 0:
                print(i)
        self.res = res
        sd_estimate = np.sqrt(np.mean(res, 2)[:, 1])
        sd_true = np.sqrt(np.var(res, 2)[:, 0])
        mean_estimate = np.mean(res, 2)[:, 0]
        mean_true = np.repeat(self.simuldata.ground_truth(), len(self.tests))
        self.report = pd.DataFrame(data={
            'Method': [test.name for test in self.tests],
            'std_estimate': sd_estimate,
            'std_true': sd_true,
            'mean_estimate': mean_estimate,
            'mean_true': mean_true
        })

    def run_once(self):
        simdf = self.simuldata.data()
        est_dict = {}
        for test in self.tests:
            est_dict[test] = test.estimate(simdf)
        est = pd.DataFrame(est_dict).transpose()
        return est

    def power_test(self):
        comparison_data = copy.deepcopy(self.simuldata)
        power_df = pd.DataFrame(columns=['test', 't', 'p', 'effect_size', 'x1', 'x2', 'delta_x', 'var1', 'var2', 'n1', 'n2'])
        for i, increment in enumerate(self.simuldata.get_increments()):
            print(f"{i}: {increment}")
            comparison_data.update_mu(increment)
            effect_size = comparison_data.ground_truth() - self.simuldata.ground_truth()
            for _ in range(100):
                original_data = self.simuldata.data()
                data = comparison_data.data()
                for test in self.tests:
                    original_estimates = test.estimate(original_data)
                    estimates = test.estimate(data)
                    (t, p) = ttest(original_estimates['Estimate'], estimates['Estimate'], original_estimates['Var']*original_data[test.level].shape[0], estimates['Var']*data[test.level].shape[0], original_data[test.level].shape[0], data[test.level].shape[0])
                    # print(f"""
                    # ======================={test.name}======================
                    # x1: {original_estimates['Estimate']}  x2:{estimates['Estimate']}
                    # var1:{original_estimates['Var']}  var2:{estimates['Var']}
                    # n1:{original_data['unitlevel'].shape[0]}  n2:{data['unitlevel'].shape[0]}
                    # t:{t}  p:{p}
                    # """)
                    power_df = power_df.append(
                        {
                            'test': test.name,
                            't': t,
                            'p': p,
                            'effect_size': f'{effect_size:.4f}',
                            'x1': original_estimates['Estimate'],
                            'x2':estimates['Estimate'],
                            'delta_x': estimates['Estimate'] - original_estimates['Estimate'],
                            'var1': original_estimates['Var']*original_data[test.level].shape[0],
                            'var2': estimates['Var']*data[test.level].shape[0],
                            'n1': original_data[test.level].shape[0],
                            'n2': data[test.level].shape[0]
                        }, ignore_index=True
                    )
        return power_df

    # def power_test(self):
    #     comparison_data = copy.deepcopy(self.simuldata)
    #     power_df = pd.DataFrame(columns=['test', 't', 'p', 'effect_size', 'x1', 'x2', 'var1', 'var2', 'n1', 'n2'])
    #     for i, increment in enumerate(self.simuldata.get_increments()):
    #         print(f"{i}: {increment}")
    #         comparison_data.update_mu(increment)
    #         effect_size = comparison_data.ground_truth() - self.simuldata.ground_truth()
    #         iterable = [(effect_size, self.simuldata.data(), comparison_data.data()) for _ in range(100)]
    #         with mp.Pool(4) as pool:
    #             rows = pool.starmap(self._get_estimates, iterable)
    #             power_df = power_df.append(list(chain.from_iterable(rows)),ignore_index=True)
    #     return power_df
    
    def _get_estimates(self, effect_size, original_data, data):
        power_rows = []
        for test in self.tests:
            original_estimates = test.estimate(original_data)
            estimates = test.estimate(data)
            (t, p) = ttest(original_estimates['Estimate'], estimates['Estimate'], original_estimates['Var']*original_data[test.level].shape[0], estimates['Var']*data[test.level].shape[0], original_data[test.level].shape[0], data[test.level].shape[0])
            # print(f"""
            # ======================={test.name}======================
            # x1: {original_estimates['Estimate']}  x2:{estimates['Estimate']}
            # var1:{original_estimates['Var']}  var2:{estimates['Var']}
            # n1:{original_data['unitlevel'].shape[0]}  n2:{data['unitlevel'].shape[0]}
            # t:{t}  p:{p}
            # """)
            power_rows.append(
                {
                    'test': test.name,
                    't': t,
                    'p': p,
                    'effect_size': f'{effect_size:.4f}',
                    'x1': original_estimates['Estimate'],
                    'x2':estimates['Estimate'],
                    'var1': original_estimates['Var']*original_data[test.level].shape[0],
                    'var2': estimates['Var']*data[test.level].shape[0],
                    'n1': original_data[test.level].shape[0],
                    'n2': data[test.level].shape[0]
                }
            )

        return power_rows

    def plot_power(self, df):

        def _power(df, alpha=.05):
            return (df['p'] < alpha).sum()/len(df['p'])

        power = df.groupby(['test', 'effect_size']).apply(_power)
        effect_size = df['effect_size'].astype('float').unique()
        legend = []
        fig = plt.figure(figsize=(12,9))
        for test in self.tests:
            plt.plot(effect_size, power[test.name])
            legend.append(test.name)
        plt.legend(legend, loc='upper left')
        plt.title(self.simuldata.summary(), fontsize=16)
        plt.suptitle(self.simuldata.name, fontweight='bold')
        plt.xlabel('Effect Size', fontsize=16)
        plt.ylabel('Power (1 - β)', fontsize=16)
        plt.show()

def ttest(x1, x2, var1, var2, n1, n2):
    sd_pooled = math.sqrt((var1 / n1) + (var2 / n2))
    try:
	    t = (x1 - x2) / sd_pooled
    except Exception:
	    t = 0
    try:
        df = (
                ((var1 / n1) + (var2 / n2)) ** 2
            ) / (
                (var1 ** 2 / ((n1 - 1) * n1 ** 2)) + var2 ** 2 / ((n2 - 1) * n2 ** 2)
            )
    except Exception:
        df = 2
    p = stats.t.sf(np.abs(t), df) * 2
    return (t, p)

if __name__ == '__main__':
    mu=[0.3, 0.5, 0.8]
    sigma=[0.05, 0.1, 0.05]
    lamb=[2, 5, 30]
    probs=[1/3, 1/2, 1/6]
    n=1000
    m=100
    # bindata=UncorBinom(1, .6, n)
    # tests=[Naive(), DeltaMethod(),GroupNoSpark()]
    # binsim=Simulation(m, bindata, tests)
    # binsim.run()
    # bindf=binsim.power_test()
    # bindf['truevar']= (.6 + bindf['effect_size'].astype(float))*(1 - .6 - bindf['effect_size'].astype(float))
    # bindf[bindf['test'].isin(['Delta Method', 'Group Method(NS)'])].groupby(['test', 'effect_size']).mean()

    hetdata = HetBinom(lamb, mu, sigma, n, probs)
    tests=[Naive(), GroupMethod(),DeltaMethod(),DeltaSpark()]
    hetsim=Simulation(m, hetdata, tests)
    hetsim.run()
    # hetdf=hetsim.power_test()
    # hetdf[hetdf['test'].isin(['Delta Method', 'Group Method(NS)'])].groupby(['test', 'effect_size']).mean()

    # mu=[0.8, 0.5, 0.3]
    # hetdata_r = HetBinom(lamb, mu, sigma, n, probs)
    # tests=[Naive(), DeltaMethod(),GroupNoSpark()]
    # hetsim_r=Simulation(m, hetdata_r, tests)
    # hetsim_r.run()
    # hetdf_r=hetsim.power_test()
    # hetdf_r[hetdf_r['test'].isin(['Delta Method', 'Group Method(NS)'])].groupby(['test', 'effect_size']).mean()

    # mu=[0.8, 0.5, 0.3]
    # hetdata_r2 = HetBinom(lamb, mu, sigma, n, probs)
    # tests=[DeltaMethod(),DeltaMethod_cluster()]
    # hetsim_r2=Simulation(m, hetdata_r2, tests)
    # hetsim_r2.run()
    # hetdf_r2=hetsim_r2.power_test()
    # hetdf_r2['delta_x'] = hetdf_r2['x2'] - hetdf_r2['x1']
    # hetdf_r2[hetdf_r2['test'].isin(['Delta Method', 'Group Method(NS)'])].groupby(['test', 'effect_size']).mean()

    # bindata2=UncorBinom(1, .6, n)
    # tests=[DeltaMethod(),DeltaMethod_cluster()]
    # binsim2=Simulation(m, bindata2, tests)
    # binsim2.run()
    # bindf2=binsim2.power_test()
    # bindf2['truevar']= (.6 + bindf2['effect_size'].astype(float))*(1 - .6 - bindf2['effect_size'].astype(float))
    # bindf2[bindf2['test'].isin(['Delta Method', 'Delta Method Cluster'])].groupby(['test', 'effect_size']).mean()

    # mu=[0.3, 0.5, 0.8]
    # sigma=[0.05, 0.1, 0.05]
    # lamb=[30, 5, 2]
    # probs=[1/3, 1/3, 1/3]
    # n=1000
    # m=100
    # hetdata2 = HetBinom(lamb, mu, sigma, n, probs)
    # tests=[Naive(), DeltaMethod(),GroupNoSpark()]
    # hetsim2=Simulation(m, hetdata2, tests)
    # hetsim2.run()
    # hetdf2=hetsim2.power_test()
    # hetdf2[hetdf2['test'].isin(['Delta Method', 'Group Method(NS)'])].groupby(['test', 'effect_size']).mean()

    # mu=[0.3, 0.5, 0.8]
    # sigma=[0.05, 0.1, 0.05]
    # lamb=[5, 5, 5]
    # probs=[1/3, 1/2, 1/6]
    # n=1000
    # m=100
    # hetdata3 = HetBinom(lamb, mu, sigma, n, probs)
    # tests=[Naive(), DeltaMethod(),GroupNoSpark()]
    # hetsim3=Simulation(m, hetdata3, tests)
    # hetsim3.run()
    # hetdf3=hetsim3.power_test()
    # hetdf3[hetdf3['test'].isin(['Delta Method', 'Group Method(NS)'])].groupby(['test', 'effect_size']).mean()

    # mu=[0.5, 0.3, 0.3]
    # sigma=[0.05, 0.1, 0.05]
    # lamb=[5, 5, 5]
    # probs=[1/3, 1/3, 1/3]
    # n=1000
    # m=100
    # hetdata4 = HetBinom(lamb, mu, sigma, n, probs)
    # tests=[Naive(), DeltaMethod(),GroupNoSpark()]
    # hetsim4=Simulation(m, hetdata4, tests)
    # hetsim4.run()
    # hetdf4=hetsim4.power_test()
    # hetdf4[hetdf4['test'].isin(['Delta Method', 'Group Method(NS)'])].groupby(['test', 'effect_size']).mean()
