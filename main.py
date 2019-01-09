from argparse import ArgumentParser
from itertools import product, chain
from time import time
from pathlib import Path
from skimage import io

from KNNTrainer import predict_knn
from myPreProcessor import preprocessImage
from features.LBP import LBP
from features.LPQ import LPQ
import numpy as np

parser = ArgumentParser()
parser.add_argument('directory', type=str)
parser.add_argument('mode', type=str, choices=['LBP', 'LPQ'])
args = parser.parse_args()

extension = '.JPG'

def read_classification_data(directory):
    for directory in Path(directory).iterdir():
        if not directory.is_dir():
            continue
        yield [[directory / folder / (img + extension) for img in ('1', '2')] for folder in
               ('1', '2', '3')], directory / ('test'+extension)


def extract_features_from_image(image_path):
    feature_vectors = []
    image = io.imread(image_path, as_gray=True)
    preprocessedImages = preprocessImage(image)
    if len(preprocessedImages) == 0:
        raise RuntimeError('Too small handwritten text at ' + str(image_path))
    for preprocessedFragment in preprocessedImages:
        if args.mode == 'LBP':
            feature_vectors.append(LBP(preprocessedFragment))
        elif args.mode == 'LPQ':
            feature_vectors.append(LPQ(preprocessedFragment))
        else:
            raise RuntimeError
    return feature_vectors


def classify(data, test):
    reference_feature_vectors_list = map(lambda images_paths: chain(*map(extract_features_from_image, images_paths)),
                                         data)
    test_feature_vectors = extract_features_from_image(test)
    scores = []
    for reference_feature_vectors in reference_feature_vectors_list:
        dissimilarity_vectors = [np.abs(x - y) for x, y in product(reference_feature_vectors, test_feature_vectors)]
        scores.append(sum(predict_knn(dissimilarity_vectors)))
    return scores.index(max(scores)) + 1


with open('results.txt', 'w') as results_file, open('time.txt', 'w') as time_file:
    for data, test in read_classification_data(args.directory):
        before = time()
        label = classify(data, test)
        after = time()
        print(label, file=results_file)
        print(after - before, file=time_file)
