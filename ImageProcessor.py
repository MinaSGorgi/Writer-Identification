from DataSetReader import readDataSet
from myPreProcessor import preprocessImage
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
        if len(writerDict[writer]) == 0:
            raise RuntimeError('Empty Image Paths')
        feature_vectors = []
        for writerImagePath in writerDict[writer]:
            image = io.imread(writerImagePath, as_gray=True)
            preprocessedImages = preprocessImage(image)
            if len(preprocessedImages) == 0:
                continue
            print(len(preprocessedImages))
            for preprocessedFragment in preprocessedImages:
                if mode == 'LBP':
                    feature_vectors.append(LBP(preprocessedFragment))
                elif mode == 'LPQ':
                    feature_vectors.append(LPQ(preprocessedFragment))
                else:
                    raise RuntimeError
        if len(feature_vectors) != 0:
            writerFeatureVectorsDict[writer] = feature_vectors
    with open('features.obj', 'wb') as features_file:
        pickle.dump(writerFeatureVectorsDict, features_file)
    return writerFeatureVectorsDict


x = processImages(Path.home() / 'Documents' / 'PatternProject' / 'iamDB', 'LPQ',25)
