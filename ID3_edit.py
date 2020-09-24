from collections import deque
import sys
import math
import random
import main
import copy
import numpy as np
import pandas as pd 


class Tree_Node:
    def __init__(self,val):
        self.val = val
        self.left = None
        self.right = None


class DTree:

    def __init__(self, filename):

        csvParser = main.csvParser(self,filename)

        self.attributeNames = csvParser.attributeNames
        self.data = csvParser.data
        self.attributes = csvParser.attributes
        self.trainingValues = csvParser.trainingValues
        self.Class = csvParser.Class

        self.root = self.ID3(self.trainingValues, self.Class, self.attributes)

    def ID3(self, trainingValues, Class, Attributes):

        if len(trainingValues) == 0:
            return None

        root = Tree_Node(-1)
        Entropy = self.getEntropy(trainingValues, Class)
        varianceImpurity = self.getVarianceImpurity(trainingValues, Class)
        root.label = self.getMajority(Class)
        if Entropy == 0 or len(Attributes) == 0:
            return root

        else:
            bestAttribute = self.chooseBestAttributeEntropy(trainingValues, Class, Attributes, Entropy)

            #Uncomment below line for split using varianceImpurity and comment the above line.

            #bestAttribute = self.chooseBestAttributeImpurity(trainingValues, Class, Attributes, varianceImpurity)

            if bestAttribute == -1:
                return root
            root.val = bestAttribute
            newAttributes = []
            for attribute in Attributes:
                if attribute != bestAttribute:
                    newAttributes.append(attribute)
            Attributes = newAttributes
            # print Attributes
            subTree = self.split(trainingValues, Class, bestAttribute)
            root.left = self.ID3(subTree[0][0], subTree[0][1], Attributes)
            root.right = self.ID3(subTree[1][0], subTree[1][1], Attributes)

            return root

    def chooseBestAttributeEntropy(self, trainingValues, Class, Attributes, Entropy):

        #why -1?
        maxInfoGain = -1
        bestAttribute = -1

        for attribute in Attributes:
                infoGain = self.getInfoGain(trainingValues, Class, Entropy, attribute)

                if infoGain > maxInfoGain:
                    maxInfoGain = infoGain
                    bestAttribute = attribute
        return bestAttribute
    def chooseBestAttributeImpurity(self, trainingValues, Class, Attributes, varianceImpurity):

        maxVarianceImpurityGain = -1
        bestAttribute = -1

        for attribute in Attributes:
                varianceImpurityGain = self.getVarianceImpurityGain(trainingValues, Class, varianceImpurity, attribute)

                if varianceImpurityGain > maxVarianceImpurityGain:
                    maxVarianceImpurityGain = varianceImpurityGain
                    bestAttribute = attribute
        return bestAttribute

    def getMajority(self, Class):

        if len(Class) == 1:
            return Class[0]

        count = 0
        for i in range(len(Class)):
            if Class[i] == 1:
                count += 1

        if count >= len(Class) / 2:
            return 1
        else:
            return 0

    def split(self, trainingValues, Class, attribute):

        trainingVal0= []
        trainingVal1 = []
        targetVal0 = []
        targetVal1 = []

        for i in range(len(trainingValues)):
            if self.data[trainingValues[i]][attribute] == 0:
                trainingVal0.append(trainingValues[i])
                targetVal0.append(Class[i])
                # print trainingVal0
                # print targetVal0
            else:
                trainingVal1.append(trainingValues[i])
                targetVal1.append(Class[i])

        return [(trainingVal0, targetVal0), (trainingVal1,targetVal1)]


    def getEntropy(self, trainingValues, Class):
        entropy=0
        values = pd.unique(Class)
        for i in values:
            fraction = Class.count(i)/len(Class)
            entropy += -fraction*np.log2(fraction)
        return entropy

    def getInfoGain(self, trainingValues, Class, Entropy, attribute):

        rows = len(trainingValues)


        subTree = self.split(trainingValues, Class, attribute)
        # print subTree
        EntropyVal0 = self.getEntropy(subTree[0][0], subTree[0][1])
        EntropyVal1 = self.getEntropy(subTree[1][0], subTree[1][1])

        probVal0 = len(subTree[0][0]) / rows
        probVal1 = 1 - probVal0

        infoGain = Entropy - (probVal0 * EntropyVal0 + probVal1 * EntropyVal1)
        return infoGain

    def getVarianceImpurity(self, trainingValues, Class):

        rows = len(trainingValues)
        # print "Variance impurty"

        pcount = 0
        ncount = 0
        count = 0
        for i in range(len(trainingValues)):
            if Class[i] == 1:
                pcount += 1
            elif Class[i] == 0:
                ncount +=	1
        # print "pcount"
        # print pcount
        # print "ncount"
        # print ncount
        # print count
        if pcount == 0 or ncount == 0:
            return 0
        pos = pcount/rows
        neg = ncount/rows
        # print pos
        # print neg
        return 	pos*neg

    def getVarianceImpurityGain(self, trainingValues, Class, varianceImpurity, attribute):

        rows = len(trainingValues)

        subTree = self.split(trainingValues, Class, attribute)
        varianceImpurityVal0 = self.getVarianceImpurity(subTree[0][0], subTree[0][1])
        varianceImpurityVal1 = self.getVarianceImpurity(subTree[1][0], subTree[1][1])


        probVal0 = len(subTree[0][0]) / rows
        probVal1 = 1 - probVal0

        varianceImpurityGain = varianceImpurity - probVal0 * varianceImpurityVal0 - probVal1 * varianceImpurityVal1
        # print "varianceImpurityGain"
        # print varianceImpurityGain
        return varianceImpurityGain


    def pruneTree(self, L, K, validation_set):

        bestTree = self.root

        accuracy = Accuracy(validation_set)
        # print accuracy
        for i in range(1, L):

            currentTree = copy.deepcopy(bestTree)

            M = random.randint(1, K)

            for j in range(1, M):

                nonLeafNodes = self.arrange(currentTree)
                # print nonLeafNodes
                N = len(nonLeafNodes)-1
                if N <= 0:
                    return bestTree

                P = random.randint(1, N)

                replaceNode = nonLeafNodes[P]
                replaceNode.val = -1
                replaceNode.left = None
                replaceNode.right = None

            oldAccuracy = accuracy.calculateAccuracy(bestTree)
            newAccuracy = accuracy.calculateAccuracy(currentTree)

            if newAccuracy >= oldAccuracy:
                bestTree = currentTree

        self.root = bestTree
        return bestTree

    def arrange(self, root):

        array = []

        if root == None or root.val == -1:
            return array

        queue = deque([root])
        while len(queue) > 0:
            currentNode = queue.popleft()
            array.append(currentNode)
            if currentNode.left != None and currentNode.left.val != -1:
                queue.append(currentNode.left)
            if currentNode.right != None and currentNode.right.val != -1:
                queue.append(currentNode.right)

        return array

    def __str__(self):
            return self.displayTree(self.root, 0, self.attributeNames)

    def displayTree(self, root, level, attributeNames):

        treeStr = ''
        if root == None:
            return ''
        if root.left == None and root.right == None:
            # print root.label
            treeStr += str(root.label) + '\n'
            return treeStr

        currentNode = attributeNames[root.val]
        # print currentNode

        depth = ''
        for i in range(0, level):
            depth += '| '
        treeStr += depth


        if root.left!=None:
            if root.left.left == None and root.left.right == None:
                treeStr +=  currentNode + "= 0 :"
            else:
                treeStr +=  currentNode + "= 0 :\n"
        treeStr += self.displayTree(root.left, level + 1, attributeNames)


        treeStr += depth
        if root.right!=None:
            if root.right.left == None and root.right.right == None:
                treeStr += currentNode + "= 1 :"
            else:
                treeStr += currentNode + "= 1 :\n"
        treeStr += self.displayTree(root.right, level + 1, attributeNames)

        return treeStr


class Accuracy:
	def __init__(self, filename):
		csvParser = main.csvParser(self,filename)
		self.Class = csvParser.Class
		self.data = csvParser.data

	def calculateAccuracy(self, root):

		if root == None or len(self.data) == 0:
			return 0
		count = 0
		for i in range(0,len(self.data)):
			if self.prediction(root, self.data[i]) == self.Class[i]:
				count += 1

		self.accuracy = count / len(self.data)
        # print self.accuracy
		return self.accuracy

	def prediction(self, root, row):
        # print root.val
        # print row[root.val]
		if root!=None:
			if root.val == -1:
				return root.label
			if row[root.val] == 0:
				return self.prediction(root.left, row)
			else:
				return self.prediction(root.right, row)

	def displayAccuracy(self):
		print (str((self.accuracy) * 100) + "%")
