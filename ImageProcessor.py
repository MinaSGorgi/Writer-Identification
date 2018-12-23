from DataSetReader import readDataSet
from preprocess import preProcessImage
from pathlib import Path
from skimage import io


def processImages():
    imagePaths = readDataSet(Path.cwd().parent / 'DataSet')
    for reader in imagePaths.keys():
        readerImages = io.imread_collection(imagePaths.get(reader), conserve_memory=False)
        for image in readerImages:
            preProcessedImage =  preProcessImage(image)


