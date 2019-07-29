import numpy as np
import statsmodels.formula.api as smf
from pyspark.sql import SparkSession
from src.calc.aggregations import _aggregate_from_day
import abc
import ipdb


class ABMethod(abc.ABC):
    name = NotImplemented  # type: str
    @abc.abstractmethod
    def estimate(self, df):
        pass


class Naive(ABMethod):
    name = "Naive"
    level = "unitlevel"

    def estimate(self, df):
        fit = smf.ols(formula="y ~ 1", data=df['unitlevel']).fit()
        coef = {
            'Estimate': fit.params.values[0],
            'Var': fit.bse.values[0]**2
        }
        return coef


class MixedModel(ABMethod):
    name = "Mixed Model"
    level = "unitlevel"

    def estimate(self, df):
        df = df['unitlevel']
        fit = smf.mixedlm("y ~ 1", groups=df["session_id"], data=df).fit()
        coef = {
            'Estimate': fit.params.values[0],
            'Var': fit.bse.values[0]**2
        }
        return coef


class GroupMethod(ABMethod):
    name = "Group Method"
    level = "clusterlevel"

    def estimate(self, df):
        # format df to match analytics
        clust = df['clusterlevel'].rename(index=str, columns={
                                    'sum': 'metric_sum',
                                    'count': 'metric_count'
                                })
        clust.reset_index(level=0, inplace=True)
        clust['partition_date_denver'] = '2019-01-03'
        clust['variant_uuid'] = 'v1'

        # build spark df
        spark = SparkSession.builder.getOrCreate()
        sparkdf = spark.createDataFrame(clust)
        ipdb.set_trace()

        # get aggregate stats
        aggregates = _aggregate_from_day(sparkdf).first()
        coef = {
            'Estimate': aggregates.mean,
            'Var': aggregates.std**2
        }
        return coef


class GroupNoSpark(ABMethod):
    name = "Group Method(NS)"
    level = "clusterlevel"

    def estimate(self, df):
        # format df to match analytics
        clust = df['clusterlevel'].rename(index=str, columns={
                                    'sum': 'metric_sum',
                                    'count': 'metric_count'
                                })
        clust.reset_index(level=0, inplace=True)
        clust['partition_date_denver'] = '2019-01-03'
        clust['variant_uuid'] = 'v1'
        clust['value'] = clust['metric_sum'] / clust['metric_count']

        coef = {
            'Estimate': np.mean(clust['value']),
            'Var': np.var(clust['value'], ddof=1)/df['clusterlevel'].shape[0]
        }
        return coef

class DeltaMethod(ABMethod):
    name = "Delta Method"
    level = "unitlevel"

    def estimate(self, df):
        clusterdf=df['clusterlevel']
        coef={
            'Estimate': clusterdf['sum'].sum() / clusterdf['count'].sum(),
            'Var': self.varRatio(clusterdf['sum'], clusterdf['count'])
        }
        return coef

    def varRatio(self, num, denom):
        mux=denom.mean()
        muy=num.mean()
        varx=denom.var()
        vary=num.var()
        covxy=np.cov(num, denom)[0][1]
        est_var=(1 / mux**2 * vary + muy**2 / mux**4 * varx - 2 * muy / mux**3 * covxy) / len(num)
        return est_var

class DeltaSpark(ABMethod):
    name = "Delta Spark"
    level = "unitlevel"
    
    def estimate(self, df):
        # format df to match analytics
        clust=df['clusterlevel'].rename(index=str, columns={
                                    'sum': 'metric_sum',
                                    'count': 'metric_count'
                                })
        clust.reset_index(level=0, inplace=True)
        clust['partition_date_denver']='2019-01-03'
        clust['variant_uuid']='v1'

        # build spark df
        spark=SparkSession.builder.getOrCreate()
        sparkdf=spark.createDataFrame(clust)
        aggregates=_aggregate_from_day(sparkdf).first()
        coef={
            'Estimate': aggregates.mean,
            'Var': aggregates.std**2/df['unitlevel'].shape[0]
        }
        return coef