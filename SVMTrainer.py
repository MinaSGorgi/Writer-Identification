from sklearn import svm
from PairWiseVectorsGenerator import getPairWise


def convert_pair_wise_vectors_to_dissemelarity_space(positive_list, negative_list):
    positive_dissimilarity_vector = list(abs(x - y) for x, y in positive_list)
    positive_labels = [1] * len(positive_dissimilarity_vector)
    negative_dissimilarity_vector = list(abs(x - y) for x, y in negative_list)
    negative_labels = [0] * len(negative_dissimilarity_vector)
    dissimilarity_vector = positive_dissimilarity_vector + negative_dissimilarity_vector
    labels = positive_labels + negative_labels
    return dissimilarity_vector, labels


def train_svm(probability=False):
    positive_list, negative_list = getPairWise(50, 50)
    dissimilarity_vector, labels = convert_pair_wise_vectors_to_dissemelarity_space(positive_list, negative_list)
    clf = svm.SVC(gamma='scale', cache_size=200, probability=probability)
    clf.fit(dissimilarity_vector, labels)
    predictions = clf.predict(dissimilarity_vector)
    return clf, predictions


if __name__ == '__main__':
    clf, predictions = train_svm()