import math
import random
import statistics

def Foldcrossvalid(begin, end, dataset):

    trainingDatas = [];
    testDatas = [];
    store = [];
    length = len(dataset);
    random.shuffle(dataset);
    beginIndex = int(begin * length);
    endIndex = int(end * length);  

    for i in range(length):
        store = dataset[i].split(',');
        store[4] = store[4].rstrip();
        if i >= beginIndex and i < endIndex:
            for i in range(len(store)):
                if i != 4:
                    store[i] = float(store[i]);
            testDatas.append(store);
        else:
            for i in range(len(store)):
                if i != 4:
                    store[i] = float(store[i]);
            trainingDatas.append(store);
        store = [];

    return trainingDatas, testDatas

def calculateDistance(testData, trainingData):
	distance = 0
	for x in range(4):
		distance = distance + pow((testData[x] - trainingData[x]), 2)
	return math.sqrt(distance)


def findKNN(trainingDatas, testData, k):

    distances = [];
    distance = 0;
    for trainingData in trainingDatas:
        for i in range(4):
            distance = calculateDistance(testData, trainingData);
        distances.append((trainingData, distance));
    
    neighbours = [];
    distances.sort(key=lambda elem: elem[1]);

    for i in range(k):
        x = distances[i][0];
        neighbours.append(x);
    return neighbours

def prediction(neighbours):
    classVote = {};
    for i in range(len(neighbours)):
        predictedClass = neighbours[i][4];
        classVote[predictedClass] = classVote.get(predictedClass, 0) + 1;

    sortedVote = sorted(classVote.items(), key = lambda item: item[1], reverse = True);
	
    return sortedVote[0][0];



if __name__ == "__main__":
    
    filename = "iris.data"
    f = open(filename, 'r')
    dataset = f.readlines();

    k = 5

    print('K = ', k);

    for it in range(10):
        print(it + 1, " time itertion: ")
        print("==============================")
        FFCBegin = 0;

        iterationAc = [];

        for i in range(k):
            trainingDatas, testDatas = Foldcrossvalid(FFCBegin, FFCBegin + 0.2, dataset);
            FFCBegin += 0.2;
            predicteds = [];
            for testData in testDatas:
                neighbouts = findKNN(trainingDatas, testData, k);
                predicted = prediction(neighbouts);
                predicteds.append(predicted);

            rightClass = [];
            for testData in testDatas:
                rightClass.append(testData[4]);

            correct = 0;
            for j in range(len(testDatas)):
                if testDatas[j][4] == predicteds[j]:
                    correct += 1;
            accuracy = float(correct/(len(testDatas))) * 100;

            print(i + 1, " time flod:");
            print(accuracy);
            iterationAc.append(accuracy);

        print(it + 1, " time itertion's means: ", statistics.mean(iterationAc));
        print(it + 1, " time itertion's STD: ", statistics.stdev(iterationAc));
        print("=============================")
