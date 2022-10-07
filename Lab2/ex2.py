import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

np.random.seed(1)
total = 10000

latency = stats.expon.rvs(0, 1/4, total)

serv1 = stats.gamma.rvs(4, 0, 1/3, int(0.25*total))
serv2 = stats.gamma.rvs(4, 0, 1/2, int(0.25*total))
serv3 = stats.gamma.rvs(5, 0, 1/2, int(0.30*total))
serv4 = stats.gamma.rvs(5, 0, 1/3, int(0.20*total))

serv = np.concatenate((serv1, serv2, serv3, serv4))

time = latency + serv

az.plot_posterior({'Time':time})
plt.show()