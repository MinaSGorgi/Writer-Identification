from PairWiseVectorsGenerator import getPairWise

import numpy as np
import random
from collections import OrderedDict

import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset


class Classifier(nn.Module):
    """
    TODO: add documentation here
    """
    def __init__(self, input_size, hidden_size, num_classes, do_prob=0.5):
        super().__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.hl = nn.Linear(hidden_size, num_classes)
        self.do = nn.Dropout(do_prob) 
        self.ru = nn.ReLU()
        self.sm = nn.LogSoftmax(dim=1)  
    
    def forward(self, x):
        return self.sm(self.ru(self.do(self.hl(self.fc(x)))))


class ListDataset(Dataset):
    def __init__(self, data_list):
        super().__init__()
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]


def init_loaders():
    num_positive = 500
    num_negative = 500
    positive_features, negative_features = getPairWise(num_positive, num_negative)

    shuffles = 5
    dataset_list = []
    for positive_feature in positive_features:
        dataset_list.append((np.concatenate(positive_feature), 1))
    for negative_feature in negative_features:
        dataset_list.append((np.concatenate(negative_feature), 0))
    for __ in range(shuffles):
        random.shuffle(dataset_list)

    pivot = int(0.9 * len(dataset_list))
    train_dataset = ListDataset(dataset_list[:pivot])
    valid_dataset = ListDataset(dataset_list[pivot:])

    batch_size = 8
    shuffle = True
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size//2, shuffle=shuffle)

    return train_loader, valid_loader
    

def check_device(use_gpu=True):
    """
    TODO: add documentation here
    """
    if not use_gpu:
        print('Usage of gpu is not allowed! Using cpu instead ...')
        device = torch.device('cpu')
    elif not torch.cuda.is_available():
        print('No support for CUDA available! Using cpu instead ...')
        device = torch.device('cpu')
    else:
        print('Support for CUDA available! Using gpu ...')
        device = torch.device('cuda')

    return device


# Util function for printing stats
def print_stat(stat, char='-', num=80, debug=True):
    if debug:
        sep = char * num
        print(sep + "\n" + stat + "\n" + sep)


def train(debug=True):
    """
    TODO: add documentation here
    """
    train_loader, valid_loader = init_loaders()

    use_gpu = True
    device = check_device(use_gpu)

    len_features = 510
    hidden_size = 205
    num_classes = 2
    model = Classifier(len_features, hidden_size, num_classes)

    starting_lr = 0.1
    scheduler_step = 1
    scheduler_gamma = 0.75
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.fc.parameters(), lr=starting_lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    model.to(device)

    epochs = 5
    validate_every = 5
    accuracy_max = 0
    print('Starting training ...')
    for epoch in range(epochs):
        step = 0
        print_stat("Ecpoch: {}/{}".format(epoch+1, epochs), char='|', debug=debug)
        
        for features, labels in train_loader:
            step += 1
            if debug:
                print("Step: {}".format(step))

            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            logps = model(features.float())
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
                    
            if step % validate_every == 0:
                model.eval()
                accuracy = 0
                
                for features, labels in valid_loader:
                    features, labels = features.to(device), labels.to(device)
                    
                    logps = model(features.float())
                            
                    ps = torch.exp(logps)
                    __, top_class = ps.topk(1, dim=1)
                    #print(top_class, labels)
                    equality = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equality.type(torch.FloatTensor))
                    valid_accuracy = accuracy / len(valid_loader) * 100
                    
                print_stat("Valid accuracy: {:.5f}%".format(valid_accuracy), debug=debug)
                
                model.train()
                
                if valid_accuracy > accuracy_max:
                    accuracy_max = valid_accuracy

                    print_stat("Saving highest accuracy: {:.5f}%".format(accuracy_max),
                                char='*', debug=True)
                    torch.save(model.state_dict(), 'model.pt')
        
        scheduler.step()


if __name__ == "__main__":
    train(debug=False)