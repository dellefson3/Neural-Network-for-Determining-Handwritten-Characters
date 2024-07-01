import math
import random

class LayerClass:
    pass
    numNodesIn = 0
    numNodesOut = 0
 
    weights = []
    biases = []
    
    costGradientW = []
    costGradientB = []
    
    weightVelocities = []
    biasVelocities = []
 
    def Layer(self, numIn, numOut):
        self.numNodesIn = numIn
        self.numNodesOut = numOut
    
        self.weights = [0] * (self.numNodesIn * self.numNodesOut)
        self.costGradientW = [0] * len(self.weights)
        self.biases = [0] * self.numNodesOut
        self.costGradientB = [0] * len(self.biases)
    
        self.weightVelocities = [0] * len(self.weights)
        self.biasVelocities = [0] * len(self.biases)
    
        self.InitializeRandomWeights()
 
 
    def CalculateOutputs1(self, inputs):
        weightedInputs = [0] * self.numNodesOut
        nodeOut = 0
        while(nodeOut < self.numNodesOut):
            weightedInput = self.biases[nodeOut]
            nodeIn = 0
            while(nodeIn < self.numNodesIn):
                weightedInput += inputs[nodeIn] * self.GetWeight(nodeIn, nodeOut)
                nodeIn += 1
            weightedInputs[nodeOut] = weightedInput
            nodeOut += 1
   
        activations = [0] * self.numNodesOut
        outputNode = 0
        while(outputNode < self.numNodesOut):
            activations[outputNode] = self.Activation(weightedInputs, outputNode)
            outputNode += 1
        return activations
 
    def CalculateOutputs2(self, inputs, learnData):
        learnData.inputs = inputs
        nodeOut = 0
        while(nodeOut < self.numNodesOut):
            weightedInput = self.biases[nodeOut]
            nodeIn = 0
            while(nodeIn < self.numNodesIn):
                weightedInput += inputs[nodeIn] * self.GetWeight(nodeIn, nodeOut)
                nodeIn += 1
            learnData.weightedInputs[nodeOut] = weightedInput
            nodeOut += 1
 
        i = 0
        while(i < len(learnData.activations)):
            learnData.activations[i] = self.Activation(learnData.weightedInputs, i)
            i += 1
        return learnData.activations
 
    def ApplyGradients(self, learnRate, regularization, momentum):
        weightDecay = (1 - regularization * learnRate)
        i = 0
        while(i < len(self.weights)):
            weight = self.weights[i]
            velocity = self.weightVelocities[i] * momentum - self.costGradientW[i] * learnRate
            self.weightVelocities[i] = velocity
            self.weights[i] = weight * weightDecay + velocity
            self.costGradientW[i] = 0
            i += 1
 
        i = 0
        while(i < len(self.biases)):
            velocity = self.biasVelocities[i] * momentum - self.costGradientB[i] * learnRate
            self.biasVelocities[i] = velocity
            self.biases[i] += velocity
            self.costGradientB[i] = 0
            i += 1
 
    def CalculateOutputLayerNodeValues(self, layerLearnData, expectedOutputs):
        i = 0
        while(i < len(layerLearnData.nodeValues)):
            costDerivative = self.CostDerivative(layerLearnData.activations[i], expectedOutputs[i])
            activationDerivative = self.ActivationDerivative(layerLearnData.weightedInputs, i)
            layerLearnData.nodeValues[i] = costDerivative * activationDerivative
            i += 1
 
    def CalculateHiddenLayerNodeValues(self, layerLearnData, oldLayer, oldNodeValues):
        newNodeIndex = 0
        while(newNodeIndex < self.numNodesOut):
            newNodeValue = 0
            oldNodeIndex = 0
            while(oldNodeIndex < len(oldNodeValues)):
                weightedInputDerivative = oldLayer.GetWeight(newNodeIndex, oldNodeIndex)
                newNodeValue += weightedInputDerivative * oldNodeValues[oldNodeIndex]
                oldNodeIndex += 1
            newNodeValue *= self.ActivationDerivative(layerLearnData.weightedInputs, newNodeIndex)
            layerLearnData.nodeValues[newNodeIndex] = newNodeValue
            newNodeIndex += 1
 
    def UpdateGradients(self, layerLearnData):
        nodeOut = 0
        while(nodeOut < self.numNodesOut):
            nodeValue = layerLearnData.nodeValues[nodeOut]
            nodeIn = 0
            while(nodeIn < self.numNodesIn):
                derivativeCostWrtWeight = layerLearnData.inputs[nodeIn] * nodeValue
                self.costGradientW[self.GetFlatWeightIndex(nodeIn, nodeOut)] += derivativeCostWrtWeight
                nodeIn += 1
            nodeOut += 1
        nodeOut = 0
        while(nodeOut < self.numNodesOut):
            derivativeCostWrtBias = 1 * layerLearnData.nodeValues[nodeOut]
            self.costGradientB[nodeOut] += derivativeCostWrtBias
            nodeOut += 1
 
    def GetWeight(self, nodeIn, nodeOut):
        flatIndex = nodeOut * self.numNodesIn + nodeIn
        return self.weights[flatIndex]
 
    def GetFlatWeightIndex(self, inputNeuronIndex, outputNeuronIndex):
        return outputNeuronIndex * self.numNodesIn + inputNeuronIndex

    def CostDerivative(self, outputActivation, expectedOutput):
        return 2 * (outputActivation - expectedOutput)

    def Activation(self, inputs, index):
        #Sigmoid Function: 1/(1+e^-x)
        if(-inputs[index] > 700):
            return 0
        else:
            return 1.0 / (1 + math.exp(-inputs[index]))

    def ActivationDerivative(self, inputs, index):
        a = self.Activation(inputs, index)
        return a * (1 - a)
    
    def RandomInNormalDistribution(self, mean, standardDeviation):
        x1 = 1 - random.random()
        x2 = 1 - random.random()
        
        y1 = math.sqrt(-2.0 * math.log(x1)) *  math.cos(2.0 * math.pi *  x2)
        return y1 * standardDeviation + mean

    def InitializeRandomWeights(self):
        i = 0
        while(i < len(self.weights)):
            self.weights[i] = self.RandomInNormalDistribution(0, 1) / math.sqrt(self.numNodesIn)
            i += 1




