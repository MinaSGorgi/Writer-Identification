from DataSetReader import readDataSet
from preprocess import preprocessImage
from features.LBP import LBP
from features.LPQ import LPQ
from skimage import io
from pathlib import Path
import pickle


def processImages(datasetPath, mode, max_writers=float('inf')):
    writerDict = readDataSet(datasetPath)
    writerFeatureVectorsDict = {}
    for index, writer in enumerate(writerDict):
        if index == max_writers:
            break
        writerFeatureVectorsDict[writer] = []
        for writerImagePath in writerDict[writer]:
            image = io.imread(writerImagePath, as_gray=True)
            preprocessedImages = preprocessImage(image)
            for preprocessedFragment in preprocessedImages:
                if mode == 'LBP':
                    writerFeatureVectorsDict[writer].append(LBP(preprocessedFragment))
                elif mode == 'LPQ':
                    writerFeatureVectorsDict[writer].append(LPQ(preprocessedFragment))
                else:
                    raise RuntimeError
    with open('features.obj', 'wb') as features_file:
        pickle.dump(writerFeatureVectorsDict, features_file)
    return writerFeatureVectorsDict


x = processImages(Path.home() / 'Documents' / 'PatternProject' / 'iamDB', 'LPQ',25)
