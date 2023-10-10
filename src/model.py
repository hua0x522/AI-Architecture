import numpy as cp
import cupy as cp 


# neural network class
class neuralNetwork:

    # initialize the neural network
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate=0.1):
        """
        The network consists of three layers: input layer, hidden layer and output layer.
        Here defined these layers.
        :param input_nodes: dimension of input
        :param hidden_nodes: dimension of hidden nodes
        :param output_nodes: dimension of output
        :param learning_rate: the learning rate of neural network
        """
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate

        self.inputs = None          # input data
        self.hidden_outputs = None  # the output of hidden layer
        self.final_outputs = None   # the output of output layer
        self.lr = learning_rate     # learning rate
        
        # init the weight of input layers to hidden layers
        self.wih = cp.random.normal(0.0, pow(self.input_nodes, -0.5),
                                    (self.hidden_nodes, self.input_nodes))
        # init the weight of hidden layers to output layers
        self.who = cp.random.normal(0.0, pow(self.hidden_nodes, -0.5),
                                    (self.output_nodes, self.hidden_nodes))
        
        # activation function is the sigmoid function
        self.activation_function = lambda x: 1. / (1 + cp.exp(-x))


    def forward(self, input_feature):
        """
        Forward the neural network
        :param input_feature: single input image, flattened [784, ]
        """
        # convert inputs list to 2d array
        self.inputs = cp.array(input_feature, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = cp.dot(self.wih, self.inputs)
        # calculate the signals emerging from hidden layer
        self.hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = cp.dot(self.who, self.hidden_outputs)
        # calculate the signals emerging from final output layer
        self.final_outputs = self.activation_function(final_inputs)
        

    def backpropagation(self, targets_list):
        """
        Propagate backwards
        :param targets_list: output onehot code of a single image, [10, ]
        """
        targets = cp.array(targets_list, ndmin=2).T

        # loss
        loss = cp.sum(cp.square(self.final_outputs - targets)) / 2

        # output layer error is the (final_outputs - target)
        output_loss = self.final_outputs - targets

        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_loss = cp.dot(self.who.T, output_loss)

        # update the weights for the links between the hidden and output layers
        self.who -= self.lr * cp.dot((output_loss * self.final_outputs * (1.0 - self.final_outputs)),
                                     cp.transpose(self.hidden_outputs))

        # update the weights for the links between the input and hidden layers
        self.wih -= self.lr * cp.dot((hidden_loss * self.hidden_outputs * (1.0 - self.hidden_outputs)),
                                     cp.transpose(self.inputs))

        return loss
