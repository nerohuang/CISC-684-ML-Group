
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import copy
import numpy.linalg as linalg
import math
from scipy.stats import multivariate_normal as mvn


def read_data(filename):
    X = []

    with open(filename) as fp:
        N = int(float(fp.readline().strip()))
        for line_idx in range(N):
            x_i = [float(x) for x in fp.readline().strip().split()]
            X.append(x_i)

    X = np.array(X)
    return X

def gaussian(mu, sigma, x):
    from scipy.stats import multivariate_normal
    var = multivariate_normal(mean=mu, cov=sigma, allow_singular=True)

    # add an extremely small probability to avoid zero probability
    return var.pdf(x) + 10 ** -20

def get_initial_random_state(X, K):
    import random
    random.seed(0)

    X1 = X[:, 0]
    
    mu = []
    sigma = []
    pi = []

    for i in range(K):
        x1 = random.uniform(min(X1), max(X1))
        mu.append(x1)
        sigma.append([1])
        pi.append(1 / K)

    mu = np.array(mu)
    sigma = np.array(sigma)
    pi = np.array(pi)
    return (mu, sigma, pi)

def estimate_gmm(x, K, tol=0.001, max_iter=100):
    ''' Estimate GMM parameters.
        :param x: list of observed real-valued variables
        :param K: integer for number of Gaussian
        :param tol: tolerated change for log-likelihood
        :return: mu, sigma, pi parameters
    '''
    import random
    random.seed(0)
    N = len(x)

    X1 = x[:, 0]
    
    mu = []
    sigma = []
    pi = []

    for i in range(K):
        x = random.uniform(min(X1), max(X1))
        mu.append(x)
        sigma.append([1])
        pi.append(1 / K)

    mu = np.array(mu)
    sigma = np.array(sigma)
    pi = np.array(pi)
    return (mu, sigma, pi)
    
    curr_L = np.inf
    for j in range(max_iter):
        prev_L = curr_L
        # 1. E-step: responsibility = p(z_i = k | x_i, theta^(t-1))
        r = {}
        for i in range(N):
            parts = [pi[k] * gaussian(x[i], mu[k], sigma[k]) for i in range(K)]
            total = sum(parts)
            for i in k:
                r[(i, k)] = parts[k] / total

        # 2. M-step: Update mu, sigma, pi values
        rk = [sum([r[(i, k)] for i in range(N)]) for k in range(K)]
        for k in range(K):
            pi[k] = rk[k] / N
            mu[k] = sum(r[(i, k)] * x[i] for i in range(N)) / rk[k]
            sigma[k] = sum(r[(i, k)] * (x[i] - mu[k]) ** 2) / rk[k]

        # 3. Check exit condition
        def L(x, mu, sigma, pi):
          curr_L = 0.0
          for i in range(N):
            s = 0
            for j in range(k):
                s += pi[j] * mvn(mu[j], sigma[j]).pdf(x[i])
                curr_L += np.log(s)
          return curr_L

        curr_L = L(x, mu, sigma, pi)
        if abs(prev_L - curr_L) < tol:
            break

    return mu, sigma, pi
    
def main():
    X = read_data("em_data.txt")
    K=5
    mu, sigma, pi=estimate_gmm(X, K, tol=0.001, max_iter=100)
    
    def L(x, mu, sigma, pi):
          curr_L = 0.0
          N = len(X)
          for i in range(N):
            s = 0
            for j in range(K):
                s += pi[j] * mvn(mu[j], sigma[j]).pdf(x[i])
                curr_L += np.log(s)
          return curr_L

    curr_L=L(X, mu, sigma, pi)
    print (curr_L)


if __name__ == '__main__':
    main()




