from numpy import numpy as np;
from scipy.stats import multivariate_normal as mvn

def estimage_gmm(x, pis, mus, sigmas, tol = 0.01, maxIter = 100):

    n, p = x.shape;
    k = len(pis);

    llOld = 0;

    for i in range(maxIter):

        #E-step:
        ws = np.zeros((k, n));
        for j in range(len(mus)):
            for i in range(n):
                ws[j, i] = pis[j] * mvn(mus[j], sigmas[j]).pdf(x[i]);
        ws /= ws.sum(0);

        #M - step:
        pis = np.zeros(k);
        for j in range(len(mus)):
            for i in range(n):
                pis[j] += ws[j, i];
        pis /= n;

        mus = np.zeros((k, p));
        for j in range(k):
            for i in range(n):
                mus[j] += ws[j, i] * x[i];
            mus[j] /= ws[j, :].sum();
        
        sigames = np.zeros((k, p, p));
        for j in range(k):
            for i in range(n):
                ys = np.reshape(x[i] - mus[j], (2, 1));
                sigames[j] += ws[j, i] * np.dot(ys, ys.T);
            sigames[j] /= ws[j, :].sum();
        
        #Update likelihood
        llNew = 0.0;
        for i in range(n):
            s = 0;
            for j in range(k):
                s += pis[j] * mvn(mus[j], sigmas[j]).pdf(x[i]);
            llNew += np.log(s);
        
        if np.abs(llNew - llOld) < tol:
            break;
        llOld = llNew;
    
    return llNew;

k = 1;
pis = np.random.random(k);
pis /= pis.sum();
mus = np.random.random((k, 2))
sigmas = np.array([np.eye(2)] * k)

#read the data to x


