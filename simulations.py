import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import lognorm, norm
import random
import time

# set the global constants
c = 1
n_x_points = 100
n_middle_sims = 1000
n_inner_sims = 1000
n_outer_sims = 1
upper_tail = 0.95
eta = 1

# define CRRA utility function
def u(x, theta_1, theta_2):
    composite = theta_1 * x + (1 - x) * theta_2
    if eta == 1:
        return(np.log(composite))
    else:
        return(1/(1-eta) * (composite**(1-eta) - 1))

# define each of the strategies
def optimal(theta_1_samples, theta_2_samples):
    # literally just maximize utility over a grid of possible allocations
    x_values = np.linspace(0, c, n_x_points)
    eu = np.zeros(n_x_points)
    for i in range(n_x_points):
        utils = [u(x_values[i], theta_1_samples[j], theta_2_samples[j]) for j in range(len(theta_1_samples))]
        eu[i] = np.mean(utils)
        
    # having calculated the EU achieved by each allocation, pick the one that was best
    x_max = c * eu.argmax()/n_x_points
    return(x_max) 

def minimax(theta_1_samples, theta_2_samples):
    # implement the minimax allocation - allocate proportional to upper bound values
    # use percentiles of the samples to determine the upper tail
    bound_1 = np.quantile(theta_1_samples, upper_tail)
    bound_2 = np.quantile(theta_2_samples, upper_tail)

    # return the minimax choice of x
    return(bound_1/(bound_1+bound_2) * c)

def bayes(theta_1_samples, theta_2_samples):
    # use the means to determine the EV of each area
    if np.mean(theta_1_samples) >= np.mean(theta_2_samples):
        return(c)
    else:
        return(0)
    
def heuristic(theta_1_samples, theta_2_samples):
    # allocate proportional to mean value of each cause
    mean_1 = np.mean(theta_1_samples)
    mean_2 = np.mean(theta_2_samples)
    
    return(c * mean_1/(mean_1 + mean_2))

# run a simulation with a given distribution of values 
def run_inner_sims(mu=0, sigma=1, n_inner_sims=n_inner_sims, seed = outer_seed):
    # Set lognormal distribution parameters
    # these params are themselves drawn from hyperpriors, with hyperparams as arguments to the function
    mu1, mu2 = norm.rvs(loc=mu, scale=sigma, size=2)
    sigma1, sigma2 = lognorm.rvs(s=sigma, scale=np.exp(mu), size=2)

    # Generate theta_1 and theta_2 samples
    theta_1_samples = lognorm.rvs(s=sigma1, scale=np.exp(mu1), size=n_inner_sims)
    theta_2_samples = lognorm.rvs(s=sigma2, scale=np.exp(mu2), size=n_inner_sims)
    
    x_opt = optimal(theta_1_samples, theta_2_samples)
    x_minimax = minimax(theta_1_samples, theta_2_samples)
    x_bayes = bayes(theta_1_samples, theta_2_samples)
    x_heuristic = heuristic(theta_1_samples, theta_2_samples)
    
    return((x_opt, x_minimax, x_bayes, x_heuristic))

def run_middle_sim():
    # collect middle simulation results
    results = pd.DataFrame(columns = ['opt', 'minimax', 'bayes', 'heuristic'])
    for _ in range(n_middle_sims):
        sim_results = pd.DataFrame([run_inner_sims()], columns = results.columns)
        results = pd.concat([results, sim_results])

    # classify which approach was best in each inner sim
    results['gap_minimax'] = np.abs(results['opt'] - results['minimax'])
    results['gap_bayes'] = np.abs(results['opt'] - results['bayes'])
    results['gap_heuristic'] = np.abs(results['opt'] - results['heuristic'])
    results['minimax_best'] = (results['gap_minimax'] < results['gap_bayes']) & (results['gap_minimax'] < results['gap_heuristic'])
    results['bayes_best'] = (results['gap_bayes'] < results['gap_minimax']) & (results['gap_bayes'] < results['gap_heuristic'])
    results['heuristic_best'] = (results['gap_heuristic'] < results['gap_bayes']) & (results['gap_heuristic'] < results['gap_minimax'])

    # collect the key stats for each middle sim
    best_minimax = results.minimax_best.mean()
    best_bayes = results.bayes_best.mean()
    best_heuristic = results.heuristic_best.mean()
    mse_minimax = np.mean((results['opt'] - results['minimax'])**2)
    mse_bayes = np.mean((results['opt'] - results['bayes'])**2)
    mse_heuristic = np.mean((results['opt'] - results['heuristic'])**2)
    
    return((best_minimax, best_bayes, best_heuristic, mse_minimax, mse_bayes, mse_heuristic))

def run_outer_sim():
    # collect outer simulation results
    overall_results = pd.DataFrame(columns = ['best_minimax', 'best_bayes', 'best_heuristic', 
                                              'mse_minimax', 'mse_bayes', 'mse_heuristic'])
    for _ in range(n_outer_sims):
        sim_results = pd.DataFrame([run_middle_sim()], columns = overall_results.columns)
        overall_results = pd.concat([overall_results, sim_results])
    
    return(overall_results)

print(run_outer_sim())

# simulate for various levels of eta, the coefficient of risk aversion
eta_results = pd.DataFrame(columns = ['best_minimax', 'best_bayes', 'best_heuristic', 
                                      'mse_minimax', 'mse_bayes', 'mse_heuristic', 'eta'])
for e in range(0, 15, 1):
    eta = e/10
    outer_results = run_outer_sim()
    outer_results['eta'] = e/10
    eta_results = pd.concat([eta_results, outer_results])
    
eta = 1
print(eta_results)