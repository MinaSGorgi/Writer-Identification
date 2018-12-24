from sklearn import svm
from PairWiseVectorsGenerator import getPairWise
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from SVMTrainer import convert_pair_wise_vectors_to_dissemelarity_space
import pickle


knn_file = 'knn_classifier.obj'

def train_knn():
    positive_list, negative_list = getPairWise(5, 5)
    dissimilarity_vector, labels = convert_pair_wise_vectors_to_dissemelarity_space(positive_list, negative_list)
    train_data, test_data, train_labels, test_labels = train_test_split(dissimilarity_vector, labels, test_size=0.2)
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(train_data, train_labels)
    pickle.dump(classifier, knn_file)
    predictions = classifier.predict(test_data)
    return classifier, predictions, test_labels


def predict_knn(features_list):
    classifier = pickle.load(knn_file)
    results = []
    for feature in features_list:
        results.append(classifier.predict(feature))

    return results

if __name__ == '__main__':
    clf, predictions,train_labels = train_knn()
    print(sum(predictions))