from collections import Counter
from graphviz import Digraph
from math import log2
import numpy as np

from toy import ToyData
from sklearn.datasets import load_digits


class ID3DecisionTreeClassifier:

    def __init__(self, minSamplesLeaf=1, minSamplesSplit=2):

        self.__nodeCounter = 0

        # the graph to visualise the tree
        self.__dot = Digraph(comment='The Decision Tree')

        # suggested attributes of the classifier to handle training parameters
        self.__minSamplesLeaf = minSamplesLeaf
        self.__minSamplesSplit = minSamplesSplit

    # Create a new node in the tree with the suggested attributes for the visualisation.
    # It can later be added to the graph with the respective function
    def new_ID3_node(self):
        node = {'id': self.__nodeCounter, 'label': None, 'attribute': None, 'value': None, 'entropy': None, 'samples': None,
                'classCounts': None, 'nodes': {}, }

        self.__nodeCounter += 1
        return node

    # adds the node into the graph for visualisation (creates a dot-node)
    def add_node_to_graph(self, node, parentid=-1):
        nodeString = ''
        for k in node:
            if ((node[k] != None) and (k != 'nodes')):
                nodeString += "\n" + str(k) + ": " + str(node[k])

        self.__dot.node(str(node['id']), label=nodeString)
        if (parentid != -1):
            self.__dot.edge(str(parentid), str(node['id']))
            nodeString += "\n" + str(parentid) + " -> " + str(node['id'])

        print(nodeString)

        return

    # make the visualisation available
    def make_dot_data(self):
        return self.__dot

    #----------------------------------------------------------------#

    def fit(self, data, target, attributes):
        root = self.new_ID3_node()

        self.createBranch(data, target, attributes, root, -1, False)

        self.add_node_to_graph(root)
        return root

    def createBranch(self, data, target, attributes, root, parentId, final):
        totalEntropy, classCounts = self.calcInformationEntropy(target)

        if totalEntropy == 0 or final:
            root["label"] = self.majority(target)
        else:
            splitData = None
            missingAttributes = []
            splitAttributes = {"name": {}, "type": {}}

            if data.shape[1] == 1:
                splitData = self.splitData(data[:, 0], data, target, specialCase=True)
                root["attribute"] = attributes["name"][0]
                if len(splitData) < len(attributes["type"][0]):
                    missingAttributes = self.findMissing(list(splitData.keys()), attributes["type"][0])
            else:
                splitIndex, splitColumn = self.findSplit(data, target, totalEntropy)
                remainingData = np.delete(data, splitIndex, 1)
                splitData = self.splitData(splitColumn, remainingData, target)

                root["attribute"] = attributes["name"][splitIndex]
                splitAttributes["name"] = np.delete(attributes["name"], splitIndex)
                splitAttributes["type"] = np.delete(attributes["type"], splitIndex)

                if len(splitData) < len(attributes["type"][splitIndex]):
                    missingAttributes = self.findMissing(list(splitData.keys()), attributes["type"][splitIndex])

            for attribute in missingAttributes:
                attributeNode = self.new_ID3_node()
                attributeNode["value"] = attribute
                attributeNode["label"] = self.majority(target)
                attributeNode["samples"] = str(0)

                root["nodes"][attribute] = attributeNode

                self.add_node_to_graph(attributeNode, root["id"])

            for attribute in splitData:
                attributeData = splitData[attribute]["data"]
                attributeTarget = splitData[attribute]["target"]

                attributeNode = self.new_ID3_node()
                attributeNode["value"] = attribute
                root["nodes"][attribute] = attributeNode

                self.createBranch(attributeData, attributeTarget, splitAttributes, attributeNode, root["id"], data.shape[1] == 1)

        root["entropy"] = totalEntropy
        root["samples"] = target.size
        root["classCounts"] = classCounts
        self.add_node_to_graph(root, parentid=parentId)

    def splitData(self, splitColumn, remainingData, targets, specialCase=False):
        splitData = {}

        for index in range(splitColumn.size):
            attribute = splitColumn[index]
            row = remainingData[index]
            target = targets[index]

            attributeDict = splitData.setdefault(attribute, {})

            attributeData = None
            if specialCase:
                attributeData = attributeDict.setdefault("data", np.array([]))
                attributeData = np.append(attributeData, row)
            else:
                attributeData = attributeDict.setdefault("data", np.empty((0, row.size), int))
                attributeData = np.append(attributeData, [row], axis=0)

            attributeTarget = attributeDict.setdefault("target", np.array([]))
            attributeTarget = np.append(attributeTarget, target)

            splitData[attribute]["data"] = attributeData
            splitData[attribute]["target"] = attributeTarget

        return splitData

    def findSplit(self, data, target, totalEntropy):
        maxInformationGain = -1
        maxIndex = -1
        maxColumn = None

        numAttributes = data.shape[1]
        for index in range(numAttributes):
            column = data[:, index]
            subsets = self.createSubsets(column, target)

            subsetEntropy = 0
            for attribute in subsets:
                subsetAttributes = subsets[attribute]

                prob = subsetAttributes.size / data.shape[0]

                attributeEntropy, ignore = self.calcInformationEntropy(subsetAttributes)

                subsetEntropy += attributeEntropy * prob

            informationGain = totalEntropy - subsetEntropy
            if informationGain > maxInformationGain:
                maxInformationGain = informationGain
                maxIndex = index
                maxColumn = column

        return maxIndex, maxColumn

    def createSubsets(self, column, target):
        subsets = {}
        for i in range(column.size):
            columnValue = column[i]
            targetValue = target[i]
            subset = subsets.setdefault(columnValue, np.array([]))
            subsets[columnValue] = np.append(subset, targetValue)
        return subsets

    def calcInformationEntropy(self, target):
        unique, counts = np.unique(target, return_counts=True)
        classCounts = dict(zip(unique, counts))

        total = 0
        for clazz in classCounts:
            total += self.entropy(classCounts[clazz], target.size)

        return total * -1, classCounts

    def entropy(self, classCount, total):
        prob = classCount / total
        return prob * log2(prob)

    def findMissing(self, existingAttributes, allAttributes):
        missing = []
        for attribute in allAttributes:
            if not attribute in existingAttributes:
                missing.append(attribute)
        return missing

    def majority(self, target):
        unique, counts = np.unique(target, return_counts=True)
        classCounts = dict(zip(unique, counts))

        maxCount = -1
        maxClass = None

        for unique in classCounts:
            if classCounts[unique] > maxCount:
                maxClass = unique
                maxCount = classCounts[unique]

        return maxClass

    def predict(self, data, root):
        predicted = []

        for row in data:
            predicted.append(self.predictHelp(row, root))
        return predicted

    def predictHelp(self, row, root):
        if len(root["nodes"]) == 0:
            return root["label"]
        else:
            attributeIndex = root["attribute"]
            attribute = row[attributeIndex]
            try:
                nextNode = root["nodes"][attribute]
                return self.predictHelp(np.delete(row, attributeIndex, 0), nextNode)
            except:
                return "unknown attribute"
