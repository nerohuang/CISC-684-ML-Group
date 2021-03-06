import VImain
import readData
from collections import deque
import sys
import csv
import math
import random
import copy
import numpy as np
import pandas as pd 

class Tree:
    def __init__(self,val):
        self.left = None
        self.right = None
        self.val = val


class BuildTree:

    def __init__(self, filename):

        csvData = readData.readCSV(self,filename)

        self.data = csvData.data
        self.Class = csvData.Class
        self.attributes = csvData.attributes
        self.attributeNames = csvData.attributeNames
        self.trainingValues = csvData.trainingValues

        self.root = self.VIH(self.trainingValues, self.Class, self.attributes)

    def __str__(self):
        return self.displayTree(self.root, 0, self.attributeNames);

    def VIH(self, trainingValues, Class, attributes):

        if len(trainingValues) == 0:
            return None

        root = Tree(-1)
        VI = self.getVI(trainingValues, Class)
        root.label = self.getMajority(Class)
        if VI == 0 or len(attributes) == 0:
            return root

        else:

            bestAttribute = self.findBestAttribute(trainingValues, Class, attributes, VI)

            if bestAttribute == -1:
                return root
            root.val = bestAttribute
            newAttributes = []
            for attribute in attributes:
                if attribute != bestAttribute:
                    newAttributes.append(attribute)
            attributes = newAttributes
            
            subTree = self.split(trainingValues, Class, bestAttribute)
            root.left = self.VIH(subTree[0][0], subTree[0][1], attributes)
            root.right = self.VIH(subTree[1][0], subTree[1][1], attributes)

            return root


    def getVI(self, trainingValues, Class):

        rows = len(trainingValues)

        oneCount = 0
        zeroCount = 0
        
        for i in range(len(trainingValues)):
            if Class[i] == 1:
                oneCount += 1
            else:
                zeroCount +=	1
        if oneCount == 0 or zeroCount == 0:
            return 0

        return 	(oneCount/rows * 1.0) * (zeroCount/rows *1.0)

    def getGain(self, trainingValues, Class, VI, attribute):

        rows = len(trainingValues)

        subTree = self.split(trainingValues, Class, attribute)
        zeroVI = self.getVI(subTree[0][0], subTree[0][1])
        oneVI = self.getVI(subTree[1][0], subTree[1][1])


        zeroPro = 1.0 * len(subTree[0][0]) / rows
        onePro = 1.0 * len(subTree[1][1]) / rows

        return VI - zeroPro * zeroVI - onePro * oneVI

    def findBestAttribute(self, trainingValues, Class, attributes, VI):

        maxVIGain = -1
        bestAttribute = -1

        for attribute in attributes:
                VIGain = self.getGain(trainingValues, Class, VI, attribute)

                if VIGain > maxVIGain:
                    maxVIGain = VIGain
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
            else:
                trainingVal1.append(trainingValues[i])
                targetVal1.append(Class[i])

        return [(trainingVal0, targetVal0), (trainingVal1,targetVal1)]


    def pruneTree(self, L, K, validation_set):

        bestTree = self.root

        accuracy = Accuracy(validation_set)
        # print accuracy
        for i in range(1, L):

            currentTree = copy.deepcopy(bestTree)

            M = random.randint(1, K)

            for j in range(1, M):

                nonLeafNodes = self.arrange(currentTree)
                
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

    def displayTree(self, root, level, attributeNames):

        treeStr = ''
        if root == None:
            return ''
        if root.left == None and root.right == None:
            #print root.label
            treeStr += str(root.label) + '\n'
            return treeStr

        currentNode = attributeNames[root.val]
        # print currentNode

        depth = ''
        for i in range(level):
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
		csvData = readData.readCSV(self,filename)
		self.Class = csvData.Class
		self.data = csvData.data

	def calculateAccuracy(self, root):

		if root == None or len(self.data) == 0:
			return 0
		count = 0
		for i in range(0,len(self.data)):
			if self.prediction(root, self.data[i]) == self.Class[i]:
				count += 1

		self.accuracy = 1.0 * count / len(self.data)
        
		return self.accuracy

	def prediction(self, root, row):
        
		if root!=None:
			if root.val == -1:
				return root.label
			if row[root.val] == 0:
				return self.prediction(root.left, row)
			else:
				return self.prediction(root.right, row)

	def displayAccuracy(self):
		print (str((self.accuracy) * 100) + "%")