import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pymc3 as pm

df = pd.read_csv('data.csv')


X1 = df.loc[:,"educ_cat"]
X2 = df.loc[:,"momage"]

Y = df.loc[:,"ppvt"]


_, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].scatter(X2, Y, alpha= 0.4)
ax[0].set_xlabel('momage')
ax[0].set_ylabel('ppvt', rotation=0)
az.plot_kde(X2, ax=ax[1])

plt.show()

with pm.Model() as lab6_model:
    α = pm.Normal('α', mu=20, sd=10)
    β = pm.Normal('β', mu=0, sd=1)
    ε = pm.HalfCauchy('ε', 5)
    μ = pm.Deterministic('μ', α + β * X1)
    y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=Y)
