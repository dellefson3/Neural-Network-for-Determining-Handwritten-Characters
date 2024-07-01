import NeuralNetwork
from PIL import Image
import random
import pickle
import os
 
size = 20
directory = 'RenamedDataSet'
testingDirectory = 'SampleImages'
numLabels = 4
 
#Neural-Network-Creation---------------------------------------------------------------------------------------------------------------
 
neuralNetwork = NeuralNetwork
neuralNetwork.NeuralNetwork([size * size, 200, 50, numLabels])
 
#Neural-Network-Creation---------------------------------------------------------------------------------------------------------------
 
def GetInputs(filepath, isInverted):
    image = Image.open(filepath)
    inputs = [0] * size * size
    resizedImage = image.resize((size, size)).convert('L')
    resizedImageWidth, resizedImageHeight = resizedImage.size
    k = 0
    for i in range (resizedImageHeight): #traverses through height of the image
        for j in range (resizedImageWidth): #traverses through width of the image
            if(isInverted):
                inputs[k] = abs(resizedImage.getpixel((j, i)) - 255)
            else:
                inputs[k] = resizedImage.getpixel((j, i))
            k += 1
    return inputs
 
def SaveData():
    #BatchLearnDataSave.pkl
    pickle.dump( neuralNetwork.layers , open( 'LayersSave.pkl' , 'wb' ) )
    pickle.dump( neuralNetwork.layerSizes , open( 'LayerSizesSave.pkl' , 'wb' ) )
    pickle.dump( neuralNetwork.batchLearnData , open( 'BatchLearnDataSave.pkl' , 'wb' ) )
 
def LoadData():
    neuralNetwork.layers = pickle.load( open( 'LayersSave.pkl' , 'rb' ))
    neuralNetwork.layerSizes = pickle.load( open( 'LayerSizesSave.pkl' , 'rb' ))
    neuralNetwork.batchLearnData = pickle.load( open( 'BatchLearnDataSave.pkl' , 'rb' ))

def CalculateTestingAccuracy():
    numCorrect = 0
    for filename in os.scandir(testingDirectory):
       if (neuralNetwork.Classify(GetInputs(filename.path, True))[0] == filename.path[len(filename.path) - 5]):
        numCorrect += 1

    return numCorrect/os.listdir(testingDirectory)
 
def GetDataPoints():
    print("Converting Images...")
    percentDone = 0
    dataPoints = [DataPoint]*len(os.listdir(directory))
    i = 0
    for filename in os.scandir(directory):
        dataPoints[i] = DataPoint().__class__()
        dataPoints[i].inputs = GetInputs(filename.path, False)
        EONum = 0
        if(filename.path[19].isnumeric()):
            EONum = int(filename.path[19] + filename.path[20])
        else:
            EONum = int(filename.path[20])
        dataPoints[i].expectedOutputs = [0] * numLabels
        dataPoints[i].expectedOutputs[EONum] = 1
 
        if(percentDone < round((i / len(dataPoints)) * 100)):
            percentDone = round((i / len(dataPoints)) * 100)
            print(str(percentDone) + "%")
 
        i += 1
    return dataPoints
 
class DataPoint:
    inputs = []
    expectedOutputs = []
 
#Neural-Network-Training---------------------------------------------------------------------------------------------------------------
 
loadData = True
learn = False
EpochNum = 20

ShutDownAfterLearn = False

if(loadData):
    print("Loading Data...")
    LoadData()

if(learn):
    for  i in range(EpochNum):
        print("Starting!")
        ShuffledDataPoints = GetDataPoints()
        random.shuffle(ShuffledDataPoints)
        neuralNetwork.Learn(ShuffledDataPoints, 0.01, 0, 0.9)
        SaveData()
        print("Epoch Test Accuracy" + str(CalculateTestingAccuracy() * 100) + "%")
 
#Neural-Network-Training---------------------------------------------------------------------------------------------------------------
 
print("Done!")
print(neuralNetwork.Classify(GetInputs("TestImage.jpg", True)))

if(ShutDownAfterLearn):
    os.system("shutdown /s /t 1")