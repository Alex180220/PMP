import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

np.random.seed(1)
lamda1, lamda2 = 1/4, 1/6

x = stats.expon.rvs(0, lamda1, 4000)
y = stats.expon.rvs(0, lamda2, 6000)

z = np.concatenate((x, y))


az.plot_posterior({'z':z})
plt.show()