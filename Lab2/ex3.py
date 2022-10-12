import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

np.random.seed(1)
prob1, prob2 = .5, .3
number = 10


regular_coin = np.random.binomial(1,prob1, 100)
fake_coin = np.random.binomial(1,prob2, 100)  

group = np.array([regular_coin, fake_coin])

az.plot_posterior({'Regular coin':regular_coin, 'Fake coin':fake_coin})
plt.show()
