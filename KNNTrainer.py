from sklearn import svm
from PairWiseVectorsGenerator import getPairWise
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from SVMTrainer import convert_pair_wise_vectors_to_dissemelarity_space
import pickle


knn_file_name = 'knn_classifier.obj'


def train_knn():
    positive_list, negative_list = getPairWise(5, 5)
    dissimilarity_vector, labels = convert_pair_wise_vectors_to_dissemelarity_space(positive_list, negative_list)
    train_data, test_data, train_labels, test_labels = train_test_split(dissimilarity_vector, labels, test_size=0.2)
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(train_data, train_labels)
    with open(knn_file_name ,'wb') as knn_file:
        pickle.dump(classifier, knn_file)
    predictions = classifier.predict(test_data)
    return classifier, predictions, test_labels


def predict_knn(features_list):
    with open(knn_file_name, 'rb') as knn_file:
        classifier = pickle.load(knn_file)
    return_val = classifier.predict(features_list)
    return return_val

if __name__ == '__main__':
    clf, predictions, train_labels = train_knn()
    result = (np.count_nonzero(train_labels == predictions) / len(predictions))
    print(result * 100)
