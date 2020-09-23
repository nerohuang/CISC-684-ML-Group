import csv;
import sys;
import algorithm;

def main():
    test = 0;

def cvsRead(self, filename):
    self.data = []
    with open("./data_sets1/data_sets1/training_set.csv", newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter = ',')
        attribute = True;
        for row in csvreader:
            #get attribute names
            if attribute:
                self.attributeNames = row[:-1]
            else:
                self.data.append([int(i) for i in row])
            attribute = False;
    self.classVal = [row[-1] for row in self.data];
    return self;

if __name__ == "__main__":
    main();