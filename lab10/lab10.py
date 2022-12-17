import numpy as np
import pymc3 as pm
import arviz as az
import scipy.stats as stats
import matplotlib.pyplot as plt


if __name__ == '__main__':
    clusters = 2
    n_cluster = [75, 175, 250]
    n_total = sum(n_cluster)
    means = [40, 55, 57]
    std_devs = [2, 5, 5]
    mix = np.random.normal(np.repeat(means, n_cluster),
    np.repeat(std_devs, n_cluster))
    az.plot_kde(np.array(mix));

    np.savetxt('date.csv',mix, fmt = '%10.3f', delimiter=",") 

    plt.hist(mix, density=True, bins=30, alpha=0.3)
    plt.show()

    clusters = [2,3,4]
    models = []
    idatas = []

    for cluster in clusters:
        with pm.Model() as model:
            p = pm.Dirichlet('p', a=np.ones(cluster))
            means = pm.Normal('means', mu=np.linspace(mix.min(), mix.max(), cluster), sd=10, shape=cluster, transform=pm.distributions.transforms.ordered)
            sd = pm.HalfNormal('sd', sd=10)
            y = pm.NormalMixture('y', w=p, mu=means, sd=sd, observed=mix)
            idata = pm.sample(1000, tune=2000, target_accept=0.9, random_seed=123, return_inferencedata=True)
            idatas.append(idata)
            models.append(model)
