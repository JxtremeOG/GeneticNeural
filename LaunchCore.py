import numpy as np

from NeuralLayers import DenseLayer, ActivationLayers
from NeuralNetwork import NeuralNetwork
import NetworkCore
import GeneticAlgoCore

from keras.api.datasets import mnist
from keras.src.utils.numerical_utils import to_categorical

def preprocess_data(x, y, limit):
    # reshape and normalize input data
    x = x.reshape(x.shape[0], 28 * 28, 1)
    x = x.astype("float32") / 255
    # encode output which is a number in range [0,9] into a vector of size 10
    # e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    y = to_categorical(y)
    y = y.reshape(y.shape[0], 10, 1)
    return x[:limit], y[:limit]


# load MNIST from server
(xTrain, yTrain), (xTest, yTest) = mnist.load_data()
xTrain, yTrain = preprocess_data(xTrain, yTrain, 1000)
xTest, yTest = preprocess_data(xTest, yTest, 20)

population = []

for index in range(200):
    network = NeuralNetwork(
        [
            DenseLayer.DenseLayer(28 * 28, 40),
            ActivationLayers.aTanh(),
            DenseLayer.DenseLayer(40, 10),
            ActivationLayers.aTanh()
        ]
    )
    population.append(network)
    
population = GeneticAlgoCore.trainGenetically(population, GeneticAlgoCore.ErrorBasedFitness, xTrain, yTrain, generationLimit=500)

for network in population:
    print(network.fitnessScore)
    for x, y in zip(xTest, yTest):
        output = NetworkCore.predict(network.layers, x)
        # print('pred:', np.argmax(output), '\ttrue:', np.argmax(y))
        print('pred:', np.argmax(output), '\ttrue:', np.argmax(y))
        print('Error:', round(NetworkCore.MeanSquaredError(y, output), 3)*100)
        print('-'*20)
    print("Done with network")  
    input("Waiting...")
    
print("Done")