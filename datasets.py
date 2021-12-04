import numpy as np
from PIL import Image

from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler

#
# This dataset class generates the data inputs for the model. In Siamese Networks, the input is pair of images where pair might
# belong to same class or belong to different class. This class does that!
#
class SiameseMNIST(Dataset):
    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset

        # If mnist_dataset is a training dataset, this flat will be True already else False
        self.train         = mnist_dataset.train

        # mnist_dataset has sets of transformation to be done on images as a preprocessing step
        self.transform     = mnist_dataset.transform

        # if mnist_dataset passes is train dataset then following IF block will run
        if self.train:
            self.train_labels = mnist_dataset.train_labels # storing labels of each record in dataset
            self.train_data   = mnist_dataset.train_data # storing data i.e. images from each record in dataset
            self.labels_set   = set(self.train_labels.numpy()) # making a set of the labels. There are 10 labels.
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0] # converting each label to a index for easy reference when working with them
                                     for label in self.labels_set}

        else:
            # generate fix pairs for testing
            self.test_labels  = mnist_dataset.test_labels
            self.test_data    = mnist_dataset.test_data
            self.labels_set   = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29) # making the randomness still and not random anymore

            # generating positive pairs i.e. pairs with images belonging to same class
            positive_pairs = [[i, random_state.choice(self.label_to_indices[self.test_labels[i].item()]),1]
                                for i in range(0, len(self.test_data), 2)]
            
            # generating negative pairs i.e. pairs with images with different class
            negative_pairs = [[i, 
                                random_state.choice(self.label_to_indices[
                                    np.random.choice(
                                        list(self.labels_set - set([self.test_labels[i].item()]))
                                        )
                                    ]),
                            0]
                for i in range(1, len(self.test_data), 2)]

            self.test_pairs = positive_pairs + negative_pairs

    def __getitem__(self, index):
        if self.train:
            # target takes 0 or 1. if 0, pair will have different class images and if 1, pair will have same class images
            target = np.random.randint(0, 2)
            img1, label1 = self.train_data[index], self.train_labels[index].item() # taking first image of the pair

            if target == 1: # positive pair
                siamese_index = index

                # selecting a different image with same class
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            
            else: # negative pair
                siamese_label = np.random.choice(list(self.labels_set - set([label1]))) # randomly choose a label different from label1
                siamese_index = np.random.choice(self.label_to_indices[siamese_label]) # randomly 
            
            img2 = self.train_data[siamese_index]

        else:
            img1 = self.test_data[self.test_pairs[index][0]]
            img2 = self.test_data[self.test_pairs[index][1]]
            target = self.test_pairs[index][2]


        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return (img1, img2), target


    def __len__(self):
        return len(self.mnist_dataset)




class TripletMNIST(Dataset):
    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset

        self.train = mnist_dataset.train
        self.transform = mnist_dataset.transform


        if self.train:
            self.train_labels = self.mnist_dataset.train_labels
            self.train_data   = self.mnist_dataset.train_data
            self.labels_set   = set(self.train_labels.numpy()) # making a set of the labels. There are 10 labels.
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0] # converting each label to a index for easy reference when working with them
                                     for label in self.labels_set}

        else:
            # generate fix pairs for testing
            self.test_labels  = mnist_dataset.test_labels
            self.test_data    = mnist_dataset.test_data
            self.labels_set   = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29) # making the randomness still and not random anymore

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.test_labels[i].item()]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.test_data))]
            self.test_triplets = triplets

    
    def __getitem__(self, index):
        if self.train:
            img1, label1 = self.train_data[index], self.train_labels[index].item()

            positive_index = index

            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[index])

            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])

            img2 = self.train_data[positive_index]
            img3 = self.train_data[negative_index]

        else:

            img1 = self.test_data[self.test_triplets[index][0]]
            img2 = self.test_data[self.test_triplets[index][1]]
            img3 = self.test_data[self.test_triplets[index][2]]

        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')
        img3 = Image.fromarray(img3.numpy(), mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return (img1, img2, img3), []

    def __len__(self):
        return len(self.mnist_dataset)