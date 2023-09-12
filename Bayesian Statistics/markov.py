import matplotlib.pyplot as plt
import numpy as np
import pymc3 
import scipy.stats as stats

plt.style.use("ggplot")

n = 50
z = 10
alpha = 12
beta = 12
alpha_post = 22
beta_post = 52

iterations = 100000

basic_model = pymc3.Model()

if __name__ == "__main__":
    with basic_model:
        # Define our prior belief about the fairness
        # of the coin using a Beta distribution
        theta = pymc3.Beta("theta", alpha=alpha, beta=beta)
        # Define the Bernoulli likelihood function
        y = pymc3.Binomial("y", n=n, p=theta, observed=z)
        # Carry out the MCMC analysis using the Metropolis algorithm
        # Use Maximum A Posteriori (MAP) optimisation as initial value for MCMC
        start = pymc3.find_MAP()
        # Use the Metropolis algorithm (as opposed to NUTS or HMC, etc.)
        step = pymc3.Metropolis()
        # Calculate the trace
        # trace = pymc3.sample(
        # iterations, step, start, random_seed=1, progressbar=True
        # )
        
        trace = pymc3.sample(draws=2000)
        
        
        
        # Plot the posterior histogram from MCMC analysis
        bins=50
        plt.hist(
        trace["theta"], bins,
        histtype="step", normed=True,
        label="Posterior (MCMC)", color="red"
        )
        # Plot the analytic prior and posterior beta distributions
        x = np.linspace(0, 1, 100)
        plt.plot(
        x, stats.beta.pdf(x, alpha, beta),
        "--", label="Prior", color="blue"
        )
        plt.plot(
        x, stats.beta.pdf(x, alpha_post, beta_post),
        label='Posterior (Analytic)', color="green"
        )


        # Show the trace plot
    pymc3.traceplot(trace)
    plt.show()
