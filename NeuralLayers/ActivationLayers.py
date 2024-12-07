from NeuralLayers.BaseLayer import BaseLayer
import numpy as np

class ActivationLayer(BaseLayer):
    def __init__(self, activation, activationPrime):
        self.activation = activation
        self.activationPrime = activationPrime
        
    def forwardProp(self, input):
        self.input = input
        return self.activation(self.input)
    
    def backwardProp(self, outputGradient, learningRate):
        return np.multiply(outputGradient, self.activationPrime(self.input))
    
class aTanh(ActivationLayer):
    def __init__(self):
        tanh = lambda x: np.tanh(x)
        tanhPrime = lambda x: 1 - np.tanh(x) ** 2
        super().__init__(tanh, tanhPrime)