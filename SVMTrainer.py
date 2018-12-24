from sklearn import svm
from PairWiseVectorsGenerator import getPairWise
import numpy as np
from sklearn.model_selection import train_test_split

def convert_pair_wise_vectors_to_dissemelarity_space(positive_list, negative_list):
    positive_dissimilarity_vector = list(np.abs(x - y) for x, y in positive_list)
    positive_labels = [1] * len(positive_dissimilarity_vector)
    negative_dissimilarity_vector = list(np.abs(x - y) for x, y in negative_list)
    negative_labels = [0] * len(negative_dissimilarity_vector)
    dissimilarity_vector = positive_dissimilarity_vector + negative_dissimilarity_vector
    labels = positive_labels + negative_labels
    return dissimilarity_vector, labels


def train_svm(probability=False):
    positive_list, negative_list = getPairWise(5, 5)
    dissimilarity_vector, labels = convert_pair_wise_vectors_to_dissemelarity_space(positive_list, negative_list)
    train_data,test_data,train_labels ,test_labels = train_test_split(dissimilarity_vector,labels,test_size=0.2)
    clf = svm.SVC(gamma='scale', cache_size=2000,C=0.5)
    clf.fit(train_data, train_labels)
    predictions = clf.predict(test_data)
    return clf, predictions,test_labels

if __name__ == '__main__':
    clf, predictions,test_lables = train_svm()
    result = (np.count_nonzero(np.array(test_lables) == predictions) / len(predictions))
    print(np.count_nonzero(predictions))
    print(result * 100)