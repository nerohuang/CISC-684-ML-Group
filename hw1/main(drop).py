import pandas as pd;
import csv;
import sys;

def VI(data):
    k = len(data['Class']);
    k1, k0 = 0, 0;
    for i in range(k):
        if data['Class'][i] == 1:
            k1 += 1;
        else:
            k0 += 1;
    return (1.0 * k1 / k) * (1.0 * k0 / k);

def pr(data, attribute, val):
    
    count = 0;
    for i in range(len(data[attribute])):
        if data[attribute][i] == val:
            count += 1;
    if count == 0 :
        return 0;
    else:
        return (1.0 * count / len(data[attribute]))

def Gain(data, attribute):

    vi = VI(data);
    
    #for value equal to 1:

    pr_1 = pr(data, attribute, 1);
    pr_0 = pr(data, attribute, 0);
    vix_1 = VI(data[data[attribute] == 1]);
    vix_0 = VI(data[data[attribute] == 0]);

    return vi - pr_1 * vix_1 - pr_0 * vix_0;

def findAttribute(data):
    maxGain = 0;
    attNum = len(data.keys())
    for attribute in data.keys()[:attNum - 1]:
        gain = Gain(data, attribute);
        maxGain = max(maxGain, gain);
    

data = pd.read_csv("F:/master/CEPG-684 Machine Learning/CISC-684-ML-Group/hw1/data_sets1/data_sets1/training_set.csv");

print(data.keys())