from operator import index
from PIL import Image
import os

directory = 'ToBeRenamed'

dataPairs = [('A', 0), ('B', 1), ('C', 2), ('D', 3), ('E', 4), ('F', 5), ('G', 6), ('H', 7), ('I', 8), ('J', 9), ('K', 10), ('L', 11), ('M', 12), ('N', 13), ('O', 14), ('P', 15), ('Q', 16), ('R', 17), ('S', 18), ('T', 19), ('U', 20), ('V', 21), ('W', 22), ('X', 23), ('Y', 24), ('Z', 25)]

def RenameFiles(baseName, startingIndex):
    Index = startingIndex
    min = 0
    while(len(os.listdir(directory)) > 0):
        i = 0
        for filename in os.scandir(directory):
            found = False
            for a, b in dataPairs:
                if(((filename.path[12] == a) and (not found))):
                    fileIndex = b
                    found = True
            if(not found):
                print("Not Labeled Correctly")
                return

            if (fileIndex == min):
                iNum = str(i)
                iIndex = str(Index)
                while(len(iNum) < 3):
                    iNum = "0" + iNum
                while(len(iIndex) < 3):
                    iIndex = "0" + iIndex
                os.rename(filename.path, "RenamedDataSet\\" + baseName + iIndex + "-" + iNum +".jpg")
                i += 1
        min += 1
        Index += 1

RenameFiles("Img", 0)