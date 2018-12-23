from DataSetReader import readDataSet
from preprocess import preprocessImage
from features.LBP import LBP
from features.LPQ import LPQ
from skimage import io
from pathlib import Path
import pickle


def processImages(datasetPath, mode, n_max=float('inf')):
    feature_vectors = []
    labels = []
    writerDict = readDataSet(datasetPath)
    for index, writer in enumerate(writerDict):
        if index == n_max:
            break
        for writerImagePath in writerDict[writer]:
            image = io.imread(writerImagePath, as_gray=True)
            preprocessedImages = preprocessImage(image)
            for preprocessedFragment in preprocessedImages:
                if mode == 'LBP':
                    feature_vectors.append(LBP(preprocessedFragment))
                elif mode == 'LPQ':
                    feature_vectors.append(LPQ(preprocessedFragment))
                else:
                    raise RuntimeError
                labels.append(writer)
    with open('features.obj', 'wb') as features_file:
        pickle.dump((feature_vectors, labels), features_file)


processImages(Path.home() / 'Documents' / 'PatternProject' / 'iamDB', 'LPQ', 5)
with open('features.obj', 'rb') as features_file:
    feature_vectors, labels = pickle.load(features_file)
