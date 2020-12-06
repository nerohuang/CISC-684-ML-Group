import numpy as np
import math
import random


def calculatePr(x, mu, sigma):
    return np.exp((((x - mu)/sigma)**2)/-2)/(sigma*math.sqrt(2*math.pi));

def calculateCluster(x, mu, sigma, alpha, k):
    newMu = [];
    newSigma = [];
    newAlpha = [];
    for i in range(k):
        #E-step:
        pr = calculatePr(x, mu[i], math.sqrt(sigma[i]));
        prAlpha = pr*alpha[i];
        prSum = 0;
        for j in range(len(alpha)):
            curPr = calculatePr(x, mu[j], math.sqrt(sigma[i]));
            prSum += curPr * alpha[j];
        wik = np.divide(prAlpha, prSum)

        #M-step:
        curNK = np.sum(wik);
        curAlphaK = curNK / len(x);
        curMuK = np.sum(np.multiply(wik, x))/np.sum(wik);
        curSigma = np.sum(np.multiply(wik, np.square(x - curMuK)))/curNK;

        newMu.append(curMuK);
        newAlpha.append(curAlphaK);
        newSigma.append(curSigma);
    return newMu, newAlpha, newSigma

def modifyEM(x, mu, sigma, alpha, k):
    newMu = [];
    newAlpha = [];
    for i in range(k):
        #E-step:
        pr = calculatePr(x, mu[i], sigma[i]);
        prAlpha = pr*alpha[i];
        prSum = 0;
        for j in range(len(alpha)):
            curPr = calculatePr(x, mu[j], sigma[j]);
            prSum += curPr * alpha[j];
        wik = np.divide(prAlpha, prSum)

        #M-step:
        curNK = np.sum(wik);
        curMuK = np.sum(np.multiply(wik, x))/np.sum(wik);
        curSigma = np.sum(np.multiply(wik, np.square(x - curMuK)))/curNK;

        newMu.append(curMuK);
        newSigma.append(curSigma);
    return newMu, alpha, newSigma


def calLog(x, mu, sigma, alpha):
    count = 0;
    for num in x:
        curCount = 0;
        for i in range(len(alpha)):
            curPr = math.exp((((num - mu[i])/math.sqrt(sigma[i])) ** 2)/-2) / (math.sqrt(sigma[i]) * math.sqrt(2*math.pi));
            curCount += alpha[i] * curPr;
        count += math.log(curCount);
    likelihood = count;
    return likelihood;


myFile = open("em_data.txt", "r");
store = myFile.read().split("\n")[0:-1];
x = np.array([float(num) for num in store]);

mu = [];
sigma = [];
alpha = [];

xMax = max(x);
xMin = min(x);

k = 1; #{1, 3, 5}

for i in range(k):
    mu.append(random.uniform(xMin, xMax));
    sigma.append(random.uniform(xMin, xMax));
    alpha.append(random.uniform(0, 1 - np.sum(alpha)));

alpha[0] = 1 - (np.sum(alpha) - alpha[0])

muChcek, sigmaCheck, alphaCheck = 1, 1, 1;

while muChcek > 0.000000001 or sigmaCheck > 0.000000001 or alphaCheck > 0.000000001:
    newMu, newSigma, newAlpha = calculateCluster(x, mu, sigma, alpha, k);
    muChcek = abs(sum(mu) - sum(newMu)) / sum(mu);
    sigmaCheck = abs(sum(sigma) - sum(newSigma)) / sum(sigma);
    alphaCheck = abs(sum(alpha) - sum(alpha)) / sum(alpha);
    mu = newMu;
    sigma = newSigma;
    alpha = newAlpha;

likelihood = calLog(x, mu, sigma, alpha);

print("Step 2:")
print(mu);
print(sigma);
print(alpha);
print(likelihood);

#======================step 3==============

for i in range(k):
    mu.append(random.uniform(xMin, xMax));
    alpha.append(random.uniform(0, 1 - np.sum(alpha)));

sigma = [1 for _ in range(k)]

muChcek, alphaCheck = 1, 1;

while muChcek > 0.000000001 or alphaCheck > 0.000000001:
    newMu, newSigma, newAlpha = calculateCluster(x, mu, sigma, alpha, k);
    muChcek = abs(sum(mu) - sum(newMu)) / sum(mu);
    alphaCheck = abs(sum(alpha) - sum(alpha)) / sum(alpha);
    mu = newMu;
    alpha = newAlpha;

likelihood = calLog(x, mu, sigma, alpha);

print("Step 3:")
print(mu);
print(sigma);
print(alpha);
print(likelihood);