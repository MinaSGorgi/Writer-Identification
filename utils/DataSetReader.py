from pathlib import Path
from collections import defaultdict


def readDataSet(dataSetLocation):
    writerImageDictionary = {}
    formsFileLocation = dataSetLocation / 'ascii' / 'forms.txt'
    imageLocation = dataSetLocation / 'forms'
    with formsFileLocation.open(mode='r') as formsFile:
        for line in formsFile:
            if line[0] == '#':
                continue
            splittedLine = line.split(' ')
            if splittedLine[1] in writerImageDictionary.keys():
                writerImageDictionary[splittedLine[1]].append(imageLocation / (splittedLine[0] + '.png'))
            else:
                writerImageDictionary[splittedLine[1]] = [imageLocation / (splittedLine[0] + '.png')]

    return writerImageDictionary


if __name__ == '__main__':
    mydict = readDataSet(Path.home() / 'Documents' / 'PatternProject' / 'iamDB')
    print(mydict)
