import csv

def readCSV(self, filename):
    self.data = []
    with open(filename,"r") as csvfile:
        csvreader = csv.reader(csvfile, delimiter = ',')
        count = 0
        for row in csvreader:
            # print row
            if count == 0:
                self.attributeNames = row[:-1]
            else:
                self.data.append([int(i) for i in row])
            count += 1

    self.attributes = range(len(self.attributeNames))
    self.trainingValues = range(len(self.data))
    self.Class = [row[-1] for row in self.data]
    return self