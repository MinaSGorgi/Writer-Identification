from pathlib import Path
from collections import defaultdict


def readDataSet(dataSetLocation: Path):
    writerImageDictionary = defaultdict(list)
    formsFileLocation = dataSetLocation / 'ascii' / 'forms.txt'
    imageLocation = dataSetLocation / 'forms'
    with formsFileLocation.open(mode='r') as formsFile:
        for line in formsFile:
            if line[0] == '#':
                continue
            splittedLine = line.split(' ')
            writerImageDictionary[splittedLine[1]].append(imageLocation / (splittedLine[0] + '.png'))
    return writerImageDictionary


mydict = readDataSet(Path.cwd().parent / 'DataSet')
print(mydict)
