import numpy as np

class Neural_network():
    def __init__(self,input_size,number_of_neurons,number_of_outputs):
        self.input_size = input_size
        self.number_of_neurons = number_of_neurons
        self.number_of_outputs = number_of_outputs
        # weight is initially set to be random uniform values
        self.w_i_h = np.random.uniform(-0.5, 0.5, (number_of_neurons, input_size)) # weight between input and hidden layer
        self.w_h_o = np.random.uniform(-0.5, 0.5, (number_of_outputs, number_of_neurons)) # weight between hidden and output layer
        #bias is initially set at 0
        self.b_i_h = np.zeros((number_of_neurons, 1)) # bias between input and hidden layer
        self.b_h_o = np.zeros((number_of_outputs, 1)) # bias between hidden and output layer

    def sigmoid(self,z):
        return 1.0 / (1.0 + np.exp(-z))


    def feedforward(self, img):
        z = self.b_i_h + self.w_i_h @ img
        h = self.sigmoid(z)

        z = self.b_h_o + self.w_h_o @ h
        o = self.sigmoid(z)
        return o,h
    def training(self,img,train_label,learn_rate):
        o ,h= self.feedforward(img)
        cost = (1 / (2 * self.input_size)) * np.sum(train_label - o)

        # Backward propagation,due to the derivative of the cost function is simply delta_o we are going to use that instead
        delta_o = o - train_label
        self.w_h_o += -learn_rate * delta_o @ np.transpose(h)
        self.b_h_o += -learn_rate * delta_o
        # Backpropagation hidden -> input (activation function derivative)
        delta_h = np.transpose(self.w_h_o) @ delta_o * (h * (1 - h))
        self.w_i_h += -learn_rate * delta_h @ np.transpose(img)
        self.b_i_h += -learn_rate * delta_h
    def prediction(self,img):
        o,h =self.feedforward(img)
        prediction = np.argmax(o)
        return prediction
