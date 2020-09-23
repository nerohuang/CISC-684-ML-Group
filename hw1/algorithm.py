import main;

class CreateTree:

    def __init__(self, filename):

        readData = main.cvsRead(self, filename);

        self.attributeNames = readData.attributeNames;
        self.data = readData.data;

        #self.dataLen = len(self.data);

        self.classVal = readData.classVal;
        self.VarianceImpurity = self.calVarianceImpurity(self.classVal);
        
    
    def calVarianceImpurity(self, classVal):

        pos, neg =0, 0;
        classLen = len(classVal);
        for i in range(len(classVal)):
            if classVal[i] == 1:
                pos += 1;
            else:
                neg += 0;

        return (1.0 * pos/classLen) * (1.0 * neg/classLen);