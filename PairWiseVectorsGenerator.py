from itertools import combinations, product
from random import choice
import pickle


def getPairWise(R, S, max_writers=float('inf')):
    positive_list = []
    negative_list = []
    with open('features.obj', 'rb') as features_file:
        writerFeatureVectorsDict = pickle.load(features_file)
    for index, writer in enumerate(writerFeatureVectorsDict):
        if index == max_writers:
            break
        V = writerFeatureVectorsDict[writer][:R]
        positive_list += combinations(V, 2)
        Q = []
        while len(Q) < S:
            key = writer
            while key == writer:
                key = choice(list(writerFeatureVectorsDict.keys()))
            Q.append(choice(writerFeatureVectorsDict[key]))
        negative_list += product(V, Q)
    return positive_list, negative_list


positive_list, negative_list = getPairWise(2, 2)

