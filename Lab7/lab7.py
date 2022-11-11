import arviz as az
import matplotlib.pyplot as plt
import math

import numpy as np
import pymc3 as pm
import pandas as pd

if __name__ == "__main__":

    data = pd.read_csv('Prices.csv')

    price = data['Price'].values
    speed = data['Speed'].values
    hardDrive = data['HardDrive'].values
    ram = data['Ram'].values
    premium = data['Premium'].values

    fig, axes = plt.subplots(2, 2, sharex=False, figsize=(10, 8))
    axes[0,0].scatter(speed, price, alpha=0.6)
    axes[0,1].scatter(hardDrive, price, alpha=0.6)
    axes[1,0].scatter(ram, price, alpha=0.6)
    axes[1,1].scatter(premium, price, alpha=0.6)
    axes[0,0].set_ylabel("Price")
    axes[0,0].set_xlabel("Speed")
    axes[0,1].set_xlabel("HardDrive")
    axes[1,0].set_xlabel("Ram")
    axes[1,1].set_xlabel("Premium")
    plt.savefig('price_correlations.png')

    price_model = pm.Model()

    with pm.Model() as price_model:
        α = pm.Normal('α', mu=0, sd=10)
        β1 = pm.Normal('β1', mu=0, sd=5)
        β2 = pm.Normal('β2', mu=2, sd=8)
        ε = pm.HalfCauchy('ε', 5)
        μ = pm.Deterministic('μ',α + β1 * speed + β2 * np.log(hardDrive))
        y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=price)
        step = pm.Slice()
        trace = pm.sample(2000, step=step, return_inferencedata=True, cores=4)