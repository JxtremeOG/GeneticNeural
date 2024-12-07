import numpy as np
from NeuralNetwork import NeuralNetwork
from NeuralLayers import DenseLayer, ActivationLayers
import NetworkCore

def ErrorBasedFitness(yExpect, yOutput):
    mse = np.mean(np.power(yExpect - yOutput, 4))  # Mean Squared Error
    fitness = 1 / (1 + mse)  # Inverse of error
    return fitness

def createChildLayers(parent1Layer, parent2Layer, fitnessAvg):
    mutationRate = (9 - fitnessAvg) / 9
    mutationRange = (-0.5, 0.5)
    
    child1Layer = DenseLayer.DenseLayer(parent1Layer.weights.shape[1], parent1Layer.weights.shape[0])
    child2Layer = DenseLayer.DenseLayer(parent2Layer.weights.shape[1], parent2Layer.weights.shape[0])
    
    splitPointWeights = np.random.randint(1,parent1Layer.weights.shape[1])
    if splitPointWeights != 0:
        parent1LeftWeight, parent1RightWeight = parent1Layer.splitWeights(splitPointWeights)
        parent2LeftWeight, parent2RightWeight = parent2Layer.splitWeights(splitPointWeights)
        
        child1Layer.weights = np.concatenate((parent1LeftWeight, parent2RightWeight), axis=1)
        child2Layer.weights = np.concatenate((parent2LeftWeight, parent1RightWeight), axis=1)
    else:
        child1Layer.weights = parent1Layer.weights.copy()
        child2Layer.weights = parent2Layer.weights.copy()
    
    splitPointBias = np.random.randint(0,parent1Layer.bias.shape[0])
    if splitPointBias != 0:
        parent1TopBias, parent1BottomBias = parent1Layer.splitBias(splitPointBias)
        parent2TopBias, parent2BottomBias = parent2Layer.splitBias(splitPointBias)
        
        child1Layer.bias = np.concatenate((parent1TopBias, parent2BottomBias), axis=0)
        child2Layer.bias = np.concatenate((parent2TopBias, parent1BottomBias), axis=0)
    else:
        child1Layer.bias = parent1Layer.bias.copy()
        child2Layer.bias = parent2Layer.bias.copy()
    
    child1Layer.mutateWeights(mutationRate=mutationRate, mutationRange=mutationRange)
    child2Layer.mutateWeights(mutationRate=mutationRate, mutationRange=mutationRange)

    child1Layer.mutateBias(mutationRate=mutationRate, mutationRange=mutationRange)
    child2Layer.mutateBias(mutationRate=mutationRate, mutationRange=mutationRange)
    
    return child1Layer, child2Layer

def trainGenetically(population: list[NeuralNetwork], fitness, xTrain, yTrain, generationLimit = 100):    
    for genIndex in range(generationLimit):
        for network in population:
            network.fitnessScore = 0
        
        count = 0
        while count < 9:
            for network in population:
                output = NetworkCore.predict(network.layers, xTrain[count % len(xTrain)])
                network.fitnessScore += fitness(yTrain[count % len(yTrain)], output)
            count+=1
                
        population = sorted(
            population,
            key=lambda network: network.fitnessScore,
            reverse=True
        )
            
        population = population[:len(population) // 10] #Should keep top 10% of networks
            
        # Print progress
        print(f"Generation {genIndex + 1} complete. "
            f"Top Fitness Score: {population[0].fitnessScore:.5f}. "
            f"Average Fitness: {np.mean([net.fitnessScore for net in population]):.5f}")
        
        newPopulation = []
        while len(newPopulation) < len(population)*9: #We want to create 90% of the new population
            parent1 = np.random.choice(population)
            parent2 = np.random.choice(population)
            child1Layers = []
            child2Layers = []
            
            for layerIndex in range(len(parent1.layers)):
                if isinstance(parent1.layers[layerIndex], ActivationLayers.aTanh):
                    continue
                child1Layer, child2Layer = createChildLayers(parent1.layers[layerIndex], parent2.layers[layerIndex], np.average((parent1.fitnessScore, parent2.fitnessScore)))
                
                child1Layers.append(child1Layer)
                child2Layers.append(child2Layer)
                
                child1Layers.append(ActivationLayers.aTanh())
                child2Layers.append(ActivationLayers.aTanh())
                
            child1 = NeuralNetwork(child1Layers)
            child2 = NeuralNetwork(child2Layers)
                
            newPopulation.append(child1)
            newPopulation.append(child2)
            
        newPopulation.extend(population)
        population = newPopulation
        
    return population