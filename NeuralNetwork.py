import Layer

layers = []
layerSizes = []
batchLearnData = []

def NeuralNetwork(paramLayerSizes):
    global layerSizes, layers
    layerSizes = paramLayerSizes
    layers = [Layer.LayerClass()] * (len(layerSizes) - 1)
    i = 0
    while(i < len(layers)):
        layers[i] = Layer.LayerClass().__class__()
        layers[i].Layer(layerSizes[i], layerSizes[i + 1])
        i += 1
    
def Classify(inputs):
    outputs = CalculateOutputs(inputs)
    predictedClass = outputs.index(max(outputs))
    return (predictedClass, outputs)
    
def CalculateOutputs(inputs):
    for layer in layers:
        inputs = layer.CalculateOutputs1(inputs)
    return inputs
    
def Learn(trainingData, learnRate, regularization, momentum):
    global batchLearnData
    print("Learning...")
    if((batchLearnData == None) or (len(batchLearnData) != len(trainingData))):
        batchLearnData = [NetworkLearnData]*len(trainingData)
        i = 0
        while(i < len(batchLearnData)):
            batchLearnData[i] = NetworkLearnData(layers)
            i += 1
    
    i = 0
    print("Updating Gradients...")
    percentDone = 0
    while(i < len(trainingData)):
        UpdateGradients(trainingData[i], batchLearnData[i])
        if(percentDone < round((i / len(trainingData)) * 100)):
            percentDone = round((i / len(trainingData)) * 100)
            print(str(percentDone) + "%")
        i += 1
    
    i = 0
    print("Applying Gradients...")
    percentDone = 0
    while(i < len(layers)):
        layers[i].ApplyGradients(learnRate / len(trainingData), regularization, momentum)
        if(percentDone < round((i / len(layers)) * 100)):
            percentDone = round((i / len(layers)) * 100)
            print(str(percentDone) + "%")
        i += 1
    
def UpdateGradients(data, learnData):
    inputsToNextLayer = data.inputs
    i = 0
    while(i < len(layers)):
        inputsToNextLayer = layers[i].CalculateOutputs2(inputsToNextLayer, learnData.layerData[i])
        i += 1
        
    outputLayerIndex = len(layers) - 1
    outputLayer = layers[outputLayerIndex]
    outputLearnData = learnData.layerData[outputLayerIndex]
    
    outputLayer.CalculateOutputLayerNodeValues(outputLearnData, data.expectedOutputs)
    outputLayer.UpdateGradients(outputLearnData)
    
    i = outputLayerIndex - 1
    while(i >= 0):
        layerLearnData = learnData.layerData[i]
        hiddenLayer = layers[i]
        
        hiddenLayer.CalculateHiddenLayerNodeValues(layerLearnData, layers[i + 1], learnData.layerData[i + 1].nodeValues)
        hiddenLayer.UpdateGradients(layerLearnData)
        i -= 1
    
class LayerLearnData:
    inputs = []
    weightedInputs = []
    activations = []
    nodeValues = []
    def __init__(self, layer):
        self.weightedInputs = [0]*layer.numNodesOut
        self.activations = [0]*layer.numNodesOut
        self.nodeValues = [0]*layer.numNodesOut

class NetworkLearnData:
    layerData = []
    def __init__(self, layers):
        self.layerData = [LayerLearnData]*len(layers)
        i = 0
        while(i < len(layers)):
            self.layerData[i] = LayerLearnData(layers[i])
            i += 1



