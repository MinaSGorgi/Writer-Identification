from classifier import train, classify
from argparse import ArgumentParser
from time import time

parser = ArgumentParser()
parser.add_argument('operation', type=str, choices=['train', 'classify'])
parser.add_argument('input_directory', type=str)
args = parser.parse_args()

with open(args.directory, 'r') as directory:
    if args.operation == 'train':
        train(read_training_data(args.directory))
    elif args.operation == 'classify':
        with open('results.txt', 'w') as results_file, open('time.txt', 'w') as time_file:
            for data, test in read_classification_data(args.directory):
                before = time()
                label = classify(data, test)
                after = time()
                print(label, file=results_file)
                print(after - before, file=time_file)
    else:
        raise RuntimeError
