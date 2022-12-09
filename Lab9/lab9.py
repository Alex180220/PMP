import pymc3 as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az

if __name__ == '__main__':
    az.style.use('arviz-darkgrid')

    # Ex 2
    num_points=500
    coords_x=stats.expon.rvs(0, 5, num_points)
    coords_y=stats.expon.rvs(0, 5, num_points)

    xm=np.average(coords_x)
    ym=np.average(coords_y)
    d1 = []
    for i in range(0, num_points):
        d1.append([coords_x[i]-5, coords_y[i]-5])

    arr = np.asarray(d1)
    #np.savetxt('date.csv',d1, fmt = '%10.3f', delimiter=",") 

    data = np.loadtxt('date.csv')
    x_1 = data[:, 0]
    y_1 = data[:, 1]
    # ex 1
    # order = 5
    order = 2
    x_1p = np.vstack([x_1**i for i in range(1, order+1)])
    x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True))
    x_1p.std(axis=1, keepdims=True)
    y_1s = (y_1 - y_1.mean()) / y_1.std()
    plt.scatter(x_1s[0], y_1s)
    plt.xlabel('x')
    plt.ylabel('y')

    with pm.Model() as model_l:
        α = pm.Normal('α', mu=0, sd=1)
        β = pm.Normal('β', mu=0, sd=10)
        ε = pm.HalfNormal('ε', 5)
        μ = α + β * x_1s[0]
        y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=y_1s)
        idata_l = pm.sample(200, tune=4, return_inferencedata=True, cores=4) # 2000
    with pm.Model() as model_p:
        α = pm.Normal('α', mu=0, sd=1)
        β = pm.Normal('β', mu=0, sd=10, shape=order)
        # ex1b
        # β = pm.Normal('β', mu=0, sd=100, shape=order)
        # β = pm.Normal('β', mu=0, sd=np.array([10, 0.1, 0.1, 0.1, 0.1]), shape=order)
        ε = pm.HalfNormal('ε', 5)
        μ = α + pm.math.dot(β, x_1s)
        y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=y_1s)
        idata_p = pm.sample(200, tune=4, return_inferencedata=True, cores=4)
    with pm.Model() as model_c:
        α = pm.Normal('α', mu=0, sd=1)
        β = pm.Normal('β', mu=0, sd=10, shape=3)
        ε = pm.HalfNormal('ε', 5)
        μ = α + pm.math.dot(β, x_1s)
        y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=y_1s)
        idata_c = pm.sample(200, tune=4, return_inferencedata=True, cores=4) 
        
    x_new = np.linspace(x_1s[0].min(), x_1s[0].max(), 100)

    α_l_post = idata_l.posterior['α'].mean(("chain", "draw")).values
    β_l_post = idata_l.posterior['β'].mean(("chain", "draw")).values
    y_l_post = α_l_post + β_l_post * x_new

    plt.plot(x_new, y_l_post, 'C1', label='linear model')

    α_p_post = idata_p.posterior['α'].mean(("chain", "draw")).values
    β_p_post = idata_p.posterior['β'].mean(("chain", "draw")).values
    idx = np.argsort(x_1s[0])
    y_p_post = α_p_post + np.dot(β_p_post, x_1s)

    plt.plot(x_1s[0][idx], y_p_post[idx], 'C2', label=f'model order {order}')

    α_c_post = idata_c.posterior['α'].mean(("chain", "draw")).values
    β_c_post = idata_c.posterior['β'].mean(("chain", "draw")).values
    idx = np.argsort(x_1s[0])
    y_c_post = α_p_post + np.dot(β_c_post, x_1s)

    plt.plot(x_1s[0][idx], y_p_post[idx], 'C2', label=f'model order {order}')

    plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
    plt.legend()


    cmp_df = az.compare({'model_l':idata_l, 'model_p':idata_p, 'model_c':idata_c},method='BB-pseudo-BMA', ic="waic", scale="deviance")
