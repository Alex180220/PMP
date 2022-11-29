import arviz as az
import matplotlib.pyplot as plt
import math
from scipy import stats

import numpy as np
import pymc3 as pm
import pandas as pd

if __name__ == "__main__":

    data = pd.read_csv('Admission.csv')

    admission = data['Admission'].values
    gre = data['GRE'].values
    gpa = data['GPA'].values

    fig, axes = plt.subplots(2, 1, sharex=False, figsize=(10, 8))
    axes[0].scatter(gre, admission, alpha=0.6)
    axes[1].scatter(gpa, admission, alpha=0.6)
    axes[0].set_ylabel("Admission")
    axes[0].set_xlabel("GRE")
    axes[1].set_xlabel("GPA")
    plt.savefig('image.png')

    with pm.Model() as model_1:
        α = pm.Normal('α', mu=0, sd=10)
        β = pm.Normal('β', mu=0, sd=2, shape=len(gre))
        μ = α + pm.math.dot(gre, β)
        θ = pm.Deterministic('θ', 1 / (1 + pm.math.exp(-μ)))
        bd = pm.Deterministic('bd', -α/β[1] - β[0]/β[1] * gre[:,0])
        yl = pm.Bernoulli('yl', p=θ, observed=admission)
        idata = pm.sample(2000, target_accept=0.9, return_inferencedata=True)
    
    posterior_0 = idata.posterior.stack(samples=("chain", "draw"))

    theta = posterior_0['θ'].mean("samples")
    idx = np.argsort(gre)
    plt.plot(gre[idx], theta[idx], color='C2', lw=3)
    plt.vlines(posterior_0['bd'].mean(), 0, 1, color='k')
    bd_hpd = az.hdi(posterior_0['bd'].values)
    plt.fill_betweenx([0, 1], bd_hpd[0], bd_hpd[1], color='k', alpha=0.5)
    plt.scatter(gre, np.random.normal(gpa, 0.02),
    marker='.', color=[f'C{x}' for x in gpa])
    az.plot_hdi(gre, posterior_0['θ'].T, color='C2', smooth=False)
    locs, _ = plt.xticks()
    plt.xticks(locs, np.round(locs + gre.mean(), 1))