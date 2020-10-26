import os;
import LogRegression;
import perceptron;

class Features:

    def __init__(self, label, data):
        self.label = label
        self.wordCount = data
        self.features = []

    def addFeatures(self, data):
        self.features.append(data)

def main():

    stopwords = ['subject', 're:', 'from' , 'to' , 'cc', 'ect', 'the'];

    for line in open("stop_words_list.txt"):
        word = line.rstrip("\n");
        stopwords.append(word);


    #dataset 1
    #trainPath = "hw 2 datasets/dataset 1/train"
    #testPath = "hw 2 datasets/dataset 1/test"

    #dataset 2
    #trainPath = "hw 2 datasets/dataset 2/train"
    #testPath = "hw 2 datasets/dataset 2/test"
#
    #dataset 3
    trainPath = "hw 2 datasets/dataset 3/train"
    testPath = "hw 2 datasets/dataset 3/test"

    path = [];
    classes = ['ham', 'spam']
    docList = [];

    for root, dirs, files in os.walk(trainPath):
        if files and root[-3:] == "ham":
            docList.append(files);
            path.append(root);
        elif files and root[-4:] == "spam":
            docList.append(files);
            path.append(root);

    print("test")


    label,vocab = LogRegression.classifer(docList,classes,path,stopwords);
    weightVector = LogRegression.trainLR(label,vocab);
    accuracyLR = LogRegression.AccuracyLR(testPath, weightVector, vocab, stopwords );
    print ("Accuracy of Data: ", accuracyLR);

    weightVector = perceptron.trainLR(label,vocab);
    accuracyPC = perceptron.AccuracyLR(testPath, weightVector, vocab, stopwords );
    print ("Accuracy of Data: ", accuracyPC);


if __name__=="__main__":
    main()