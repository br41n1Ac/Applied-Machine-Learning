
class ToyData:

    def __init__(self):

        self.attributes = {'color':['y', 'g', 'b'], 'size':['s','l'], 'shape':['r', 'i']}
        self.classes = ('+', '-')

        self.data = [('y', 's', 'r'),
                 ('y', 's', 'r'),
                 ('g', 's', 'i'),
                 ('g', 'l', 'i'),
                 ('y', 'l', 'r'),
                 ('y', 's', 'r'),
                 ('y', 's', 'r'),
                 ('y', 's', 'r'),
                 ('g', 's', 'r'),
                 ('y', 'l', 'r'),
                 ('y', 'l', 'r'),
                 ('y', 'l', 'r'),
                 ('y', 'l', 'r'),
                 ('y', 'l', 'r'),
                 ('y', 's', 'i'),
                 ('y', 'l', 'i')]
        self.target = ('+', '-', '+', '-', '+', '+', '+', '+', '-', '-', '+', '-', '-', '-', '+', '+')

        self.testData = [('y', 's', 'r'),
                 ('y', 's', 'r'),
                 ('g', 's', 'i'),
                 ('b', 'l', 'i'),
                 ('y', 'l', 'r')]

        self.testTarget = ('+', '-', '+', '-', '+')

    def get_data(self):
        return self.attributes, self.classes, self.data, self.target, self.testData, self.testTarget

