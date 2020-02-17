import math
from collections import Counter, defaultdict

import numpy as np
from graphviz import Digraph


def entropy_initial(target):
    classes = {}
    ent = 0
    for type in target:
        if type not in classes:
            classes[type] = 1
        else:
            classes[type] = classes[type] + 1

    for label in classes.keys():
        p_x = classes[label] / len(target)
        ent += - p_x * math.log(p_x, 2)
    entropy = dict()
    entropy[ent] = sum(classes.values())
    return ent, sum(classes.values())


def calculateEntropy(target):
    pass


class ID3DecisionTreeClassifier:

    def __init__(self, minSamplesLeaf=1, minSamplesSplit=2):

        self.__nodeCounter = 0

        # the graph to visualise the tree
        self.__dot = Digraph(comment='The Decision Tree')

        # suggested attributes of the classifier to handle training parameters
        self.__minSamplesLeaf = minSamplesLeaf
        self.__minSamplesSplit = minSamplesSplit
        self.__originalAttributes = {}
        self.nodes = {'nodes': []}

    # Create a new node in the tree with the suggested attributes for the visualisation.
    # It can later be added to the graph with the respective function
    def new_ID3_node(self):
        node = {'id': self.__nodeCounter, 'label': None, 'attribute': None, 'entropy': None, 'samples': None,
                'classCounts': None, 'nodes': None}

        self.__nodeCounter += 1
        return node

    def new_ID3_node2(self, label, attribute, entropy, sample, classCounts, nodes, decision):
        node = {'id': self.__nodeCounter, 'label': label, 'attribute': attribute, 'entropy': entropy, 'samples': sample,
                'classCounts': classCounts, 'nodes': nodes, 'decision': decision, 'children': []}

        self.__nodeCounter += 1
        return node

    # adds the node into the graph for visualisation (creates a dot-node)
    def add_node_to_graph(self, node, parentid=-1):
        nodeString = ''
        for k in node:
            if (node[k] != None) and (k != 'nodes') and (k != 'children') and (k != 'decision'):
                nodeString += "\n" + str(k) + ": " + str(node[k])

        self.__dot.node(str(node['id']), label=nodeString)
        self.nodes['nodes'].append(node)
        if (parentid != -1):
            self.__dot.edge(str(parentid), str(node['id']))
            nodeString += "\n" + str(parentid) + " -> " + str(node['id'])

        print(nodeString)

        return

    # make the visualisation available
    def make_dot_data(self):
        return self.__dot

    # For you to fill in; Suggested function to find the best attribute to split with, given the set of
    # remaining attributes, the currently evaluated data and target.
    def find_split_attr(self):
        print('find split atri', self.__dot)
        # Change this to make some more sense
        return None

    # the entry point for the recursive ID3-algorithm, you need to fill in the calls to your recursive implementation
    def fit(self, data, target, attributes):
        # fill in something more sensible here... root should become the output of the recursive tree creation
        initial, samples = entropy_initial(target)
        self.__originalAttributes = attributes
        self.id3(data, target, attributes, None, True, None)
        return self.nodes

    def id3(self, data, target, attributes, root, first, val):
        entropy, samples = entropy_initial(target)

        if entropy == 0:
            node = self.create_node(self.get_class_counts(target).most_common(1)[0][0], None, entropy, samples,
                                    self.get_class_counts(target), root, val)
            return node

        else:
            if len(attributes) == 0:
                node = self.create_node(self.most_common(target), None, entropy, samples,
                                        self.get_class_counts(target), root, val)
                return node
            else:
                att, remaining, current_index, splitting_column = self.choose_attribute(entropy, samples, target,
                                                                                        attributes, data)
                remaindata = np.delete(data, current_index, 1)
                splitData = self.splitData(splitting_column, remaindata, target)
                if first:
                    node = self.new_ID3_node2(None, att, entropy, samples, self.get_class_counts(target), None, val)
                    self.add_node_to_graph(node)
                else:
                    node = self.create_node(None, att, entropy, samples, self.get_class_counts(target), root,
                                            val)
                for val in splitData:
                    attributeData = splitData[val]["data"]
                    attributeTarget = splitData[val]["target"]

                    new_node = self.id3(attributeData, attributeTarget, remaining, node, False, val)
                    decision = self.find_missing(new_node)
                    self.add_missing(new_node, decision)

            return node

    def create_node(self, label, attribute, entropy, sample, classCounts, root, decision):
        node = self.new_ID3_node2(label, attribute, entropy, sample, classCounts, root['id'],
                                  decision)
        self.add_children(root, node)
        self.add_node_to_graph(node, root['id'])
        return node

    def add_missing(self, parentNode, decision):
        if decision is not None:
            for path in decision:
                node_new = self.new_ID3_node2(parentNode['classCounts'].most_common(1)[0][0], None, 0.0, 0,
                                              'Counter()',
                                              parentNode['id'], path)
                self.add_children(parentNode, node_new)
                self.add_node_to_graph(node_new, parentNode['id'])

    def add_children(self, parent_node, node):
        parent_node['children'].append(node['id'])

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

    def find_missing(self, root):
        if len(root['children']) > 0 and root['attribute'] is not None:
            if len(root['children']) < len(self.__originalAttributes[root['attribute']]):
                temp = root['children']
                atts = np.array(self.__originalAttributes[root['attribute']])
                done_nodes = []
                for children in temp:
                    temp_node = self.nodes['nodes'][children]['decision']
                    done_nodes.append(temp_node)
                something = np.array(done_nodes)
                return np.setdiff1d(atts, something)

    def predict(self, data, tree, toy):
        predicted = []
        root = tree['nodes'][0]
        for row in data:
            predicted.append(self.pred(tuple(row), root, tree, toy))
        return predicted

    def pred(self, row, node, tree, toy):
        if not node['children']:
            label = node['label']
            return label
        else:
            children = node['children']
            for child in children:
                if toy:
                    if tree['nodes'][child]['decision'] in row:
                        return self.pred(row, tree['nodes'][child], tree, toy)
                else:
                    if tree['nodes'][child]['decision'] == row[node['attribute']]:
                        return self.pred(row, tree['nodes'][child], tree, toy)

    def entropy(self, data, target):

        lst = list(map(lambda x, y: (x, y), data, target))
        types = dict()
        for type in lst:
            if type[0] in types:
                types[type[0]].append(type[1])
            else:
                types[type[0]] = list()
                types[type[0]].append(type[1])

        entropy = []
        for type in types.keys():
            entropy.append(entropy_initial(types[type]))
        return entropy

    def information_gain(self, initial_entropy, samples, entropy):
        tot = samples
        initial = initial_entropy
        sum = 0
        for i in range(len(entropy)):
            sum += entropy[i][0] * entropy[i][1] / tot
        info_gain = initial - sum

        return info_gain

    def choose_attribute(self, initial_entropy, samples, target, attributes, data):
        max_info_gain = 0
        current_index = 0

        # get the index of maximum info
        for i in range(len(attributes.keys())):
            lst = [item[i] for item in data]
            temp_entropy = self.entropy(lst, target)
            temp = self.information_gain(initial_entropy, samples, temp_entropy)
            if temp is None:
                return
            if temp > max_info_gain:
                max_info_gain = temp
                current_index = i

        # get remaining attributes
        splitting_attribute = list(attributes)[current_index]
        attributes_remaining = attributes.copy()
        attributes_remaining.pop(list(attributes)[current_index])

        # get which column split
        splitting_column = []
        for row in data:
            splitting_column.append(row[current_index])
        return splitting_attribute, attributes_remaining, current_index, np.array(splitting_column)

    def get_class_counts(self, target):
        return Counter(target)

    def most_common(self, target):
        label = 0
        try:
            label = Counter(target).most_common(1)[0][0]
        except:
            print(Counter(target))
        return label
