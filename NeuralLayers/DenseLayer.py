import numpy as np
from NeuralLayers.BaseLayer import BaseLayer

class DenseLayer(BaseLayer):
    def __init__(self, inputSize, outputSize):
        self.weights = np.random.randn(outputSize, inputSize)
        self.bias = np.random.randn(outputSize, 1)
        
    def forwardProp(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias
    
    def backwardProp(self, outputGradient, learningRate):
        weightsGradient = np.dot(outputGradient, self.input.T)
        oldWeights = self.weights.copy()
        self.weights -= learningRate * weightsGradient
        self.bias -= learningRate * outputGradient
        return np.dot(oldWeights.T, outputGradient)
    
    def splitWeights(self, splitPoint):
        if not (0 < splitPoint < self.weights.shape[1]):
            raise ValueError("Split point must be within the valid range of the weights matrix columns.")
        
        leftPart = self.weights[:, :splitPoint]
        rightPart = self.weights[:, splitPoint:]
        return leftPart, rightPart
    
    def splitBias(self, splitPoint):
        if not (0 < splitPoint < self.bias.shape[0]):
            raise ValueError("Split point must be within the valid range of the bias vector rows.")
        
        topPart = self.bias[:splitPoint, :]
        bottomPart = self.bias[splitPoint:, :]
        return topPart, bottomPart
    
    def mutateWeights(self, mutationRate, mutationRange=(-0.1, 0.1)):
        if not (0 <= mutationRate <= 1):
            raise ValueError("Mutation rate must be between 0 and 1.")
        
        # Iterate through every weight in the weights matrix
        for i in range(self.weights.shape[0]):
            for j in range(self.weights.shape[1]):
                if np.random.rand() < mutationRate:
                    # Apply mutation by adding a random value within the mutation range
                    self.weights[i, j] += (self.weights[i, j] * np.random.uniform(*mutationRange))
                    
    def mutateBias(self, mutationRate, mutationRange=(-0.1, 0.1)):
        if not (0 <= mutationRate <= 1):
            raise ValueError("Mutation rate must be between 0 and 1.")
        
        for i in range(self.bias.shape[0]):
            if np.random.rand() < mutationRate:
                self.bias[i, 0] += (self.bias[i, 0] * np.random.uniform(*mutationRange))
    