import numpy as np

from NeuralLayers import DenseLayer, ActivationLayers
from NeuralNetwork import NeuralNetwork
import NetworkCore
import GeneticAlgoCore

xInputs = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
yExpected = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

# Backpropagation and typical training behavior

# network = [
#     DenseLayer.DenseLayer(2, 3),
#     ActivationLayers.aTanh(),
#     DenseLayer.DenseLayer(3, 1),
#     ActivationLayers.aTanh()
# ]
    
# NetworkCore.train(network, NetworkCore.MeanSquaredError, NetworkCore.MeanSquaredErrorPrime, xInputs, yExpected, epochs=1000, learning_rate=0.1)

# # test
# for x, y in zip(xInputs, yExpected):
#     output = NetworkCore.predict(network, x)
#     print('pred:', abs(round(np.max(output), 3)), '\ttrue:', np.max(y))
    
#Genetic Algorithm

population = []

for index in range(200):
    network = NeuralNetwork(
        [
            DenseLayer.DenseLayer(2, 3),
            ActivationLayers.aTanh(),
            DenseLayer.DenseLayer(3, 1),
            ActivationLayers.aTanh()
        ]
    )
    population.append(network)
    
population = GeneticAlgoCore.trainGenetically(population, GeneticAlgoCore.ErrorBasedFitness, xInputs, yExpected, generationLimit=1000)

for network in population:
    print(network.fitnessScore)
    for x, y in zip(xInputs, yExpected):
        output = NetworkCore.predict(network.layers, x)
        # print('pred:', np.argmax(output), '\ttrue:', np.argmax(y))
        print('Input:', x[0], x[0], 'pred:', abs(round(np.max(output), 3)), '\ttrue:', y[0][0])
        print('Error:', round(NetworkCore.MeanSquaredError(y, output), 3)*100)
    print("Done with network")
    input("Waiting...")
    
print("Done")