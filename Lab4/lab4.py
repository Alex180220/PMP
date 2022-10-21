from gc import freeze
from multiprocessing.dummy import freeze_support
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm

np.random.seed(123)


if __name__ == '__main__':
    print('')
    with pm.Model() as lab4_model:
        l = 20
        avg = 1
        std_dev = 0.5
        alpha = 5
        
        clients = pm.Poisson('poisson', mu = l)
        order = pm.Normal('order time', avg, sigma = std_dev)
        prep = pm.Exponential('preparation time', alpha)

        idata = pm.sample(100, return_inferencedata=True)
        az.plot_trace(idata);
        plt.show()
