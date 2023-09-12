import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import scipy.stats as stats

plt.style.use("ggplot")

n = 50
z = 10
alpha = 12
beta = 12
alpha_post = 22
beta_post = 52

iterations = 100000
draws = 1000

basic_model = pm.Model()

if __name__ == "__main__":
    with basic_model:
        theta = pm.Beta("theta", alpha=alpha, beta=beta)
        y = pm.Binomial("y", n=n, p=theta, observed=z)

        start = pm.find_MAP()
        step = pm.Metropolis()
        trace = pm.sample(draws, random_seed=1, progressbar=True)
    bins = 50
    plt.hist(trace["theta"], bins, histtype="step", normed=True, label="Posterior (MCMC)", color="red")

    x = np.linspace(0, 1, 100)
    plt.plot(x, stats.beta.pdf(x, alpha, beta), "--", label="Prior", color="blue")

    plt.plot(x, stats.beta.pdf(x, alpha_post, beta_post), label="Posterior (Analytic)", color="green")

    plt.legend(title="Parameters", loc="best")
    plt.xlabel("$\\theta$, Fairness")
    plt.ylabel("Density")
    plt.show()

    pm.traceplot(trace)
    plt.show()
