from collections import OrderedDict
import numpy as np


class ToyData:

    def __init__(self):
        self.attributes = {"name": np.array(["color", "size", "shape"]), "type": np.array([["y", "g", "b"], ["s", "l"], ["r", "i"]])}
        self.classes = ('+', '-')

        self.data = np.array([['y', 's', 'r'],
                              ['y', 's', 'r'],
                              ['g', 's', 'i'],
                              ['g', 'l', 'i'],
                              ['y', 'l', 'r'],
                              ['y', 's', 'r'],
                              ['y', 's', 'r'],
                              ['y', 's', 'r'],
                              ['g', 's', 'r'],
                              ['y', 'l', 'r'],
                              ['y', 'l', 'r'],
                              ['y', 'l', 'r'],
                              ['y', 'l', 'r'],
                              ['y', 'l', 'r'],
                              ['y', 's', 'i'],
                              ['y', 'l', 'i']])
        self.target = np.array(['+', '-', '+', '-', '+', '+', '+', '+', '-', '-', '+', '-', '-', '-', '+', '+'])

        self.testData = np.array([['y', 's', 'r'],
                                  ['y', 's', 'r'],
                                  ['g', 's', 'i'],
                                  ['b', 'l', 'i'],
                                  ['y', 'l', 'r']])

        self.testTarget = np.array(['+', '-', '+', '-', '+'])

    def get_data(self):
        return self.attributes, self.classes, self.data, self.target, self.testData, self.testTarget
