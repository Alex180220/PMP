import arviz as az
import matplotlib.pyplot as plt

import numpy as np

import pymc3 as pm

mu = 0
sigma = 0.05
s = np.random.normal(mu, sigma, 2000)

abs(mu - np.mean(s))

abs(sigma - np.std(s, ddof=1))

count, bins, ignored = plt.hist(s, 100, density=True)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
               np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
         linewidth=2)
plt.show()