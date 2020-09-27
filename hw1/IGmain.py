import InformationGain
import sys
from collections import defaultdict

def main():
    #directory="./data_sets1/data_sets1/"
    #directory="./data_sets2/data_sets2/"

    L = int(sys.argv[1])
    K = int(sys.argv[2])
    trainFile = sys.argv[3]
    validationFile = sys.argv[4]
    testFile = sys.argv[5]
    printJug = str(sys.argv[6])


    decisionTree = InformationGain.BuildTree(trainFile)
    if printJug == "yes":
        print ("Decision Tree by using Information Gain before Pruning:")
        print (decisionTree);

    accuracy = InformationGain.Accuracy(testFile)
    accuracy.calculateAccuracy(decisionTree.root)
    print ("Accuracy before Pruning:")
    accuracy.displayAccuracy()

    decisionTree.pruneTree(L, K, validationFile)
    if printJug == "yes":
        print ("Decision Tree by using Information Gain after Pruning:")
        print (decisionTree)
    accuracy.calculateAccuracy(decisionTree.root)
    print ("Accuracy after Pruning:")
    accuracy.displayAccuracy()

if __name__ == '__main__':
    main()