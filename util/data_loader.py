#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from itertools import compress
import random
import glob, math
import pandas as pd
import numpy as np
import pickle
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms

from matplotlib import pyplot as plt


# In[2]:


def plot_image(image, root_folder='adversarial_examples', subfolder='unnamed', filename='unnamed', 
               plot_title='', save_fig=True, show_fig=True, extension='png'):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axis('off')
    ax.set_title(plot_title)
    ax.imshow(image, interpolation='bilinear')
    
    if save_fig:
        create_dir('{}'.format(root_folder))
        create_dir('{}/{}'.format(root_folder, subfolder))
        plt.savefig('{}/{}/{}.{}'.format(root_folder, subfolder, filename, extension))
    if show_fig:
        plt.show()
    plt.clf()
    plt.close()


# In[3]:


class Dataloader:
    """
    Base class for loading data; any other dataset loading class should inherit from this to ensure consistency
    """
    TRAIN = 'train'
    TEST = 'test'
    
    def __init__(self):
        pass
    
    def length(self, portion):
        raise NotImplementedError("Implement length in child class!")

    def num_classes(self):
        raise NotImplementedError("Implement num_classes in child class!")
        
    def get_image(self, portion, idx):
        raise NotImplementedError("Implement get_image in child class!")


# In[4]:

class Adience(Dataloader):
    
    """
    Class that does the dirty loading for Adience dataset
    """
    
    def __init__(self, name, root_dir='.', base_dir=None):
        self.name = name
        self.classes = ['0-2', '4-6', '8-13', '15-20', '25-32', '38-43', '48-53', '60-']
        self.data_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        info_dir = f'{root_dir}/../data/adience'
        base_dir = info_dir if base_dir is None else f'{root_dir}/../data/{base_dir}'
        
        fold_0 = pd.read_csv('{}/fold_0_data.txt'.format(info_dir), sep='\t')
        fold_1 = pd.read_csv('{}/fold_1_data.txt'.format(info_dir), sep='\t')
        fold_2 = pd.read_csv('{}/fold_2_data.txt'.format(info_dir), sep='\t')
        fold_3 = pd.read_csv('{}/fold_3_data.txt'.format(info_dir), sep='\t')
        fold_4 = pd.read_csv('{}/fold_4_data.txt'.format(info_dir), sep='\t')

        fold_0 = fold_0[np.logical_and(fold_0['age'] != 'None', 
                           np.logical_or(fold_0['gender'] == 'm', fold_0['gender'] == 'f'))]
        fold_1 = fold_1[np.logical_and(fold_1['age'] != 'None', 
                           np.logical_or(fold_1['gender'] == 'm', fold_1['gender'] == 'f'))]
        fold_2 = fold_2[np.logical_and(fold_2['age'] != 'None', 
                           np.logical_or(fold_2['gender'] == 'm', fold_2['gender'] == 'f'))]
        fold_3 = fold_3[np.logical_and(fold_3['age'] != 'None', 
                           np.logical_or(fold_3['gender'] == 'm', fold_3['gender'] == 'f'))]
        fold_4 = fold_4[np.logical_and(fold_4['age'] != 'None', 
                           np.logical_or(fold_4['gender'] == 'm', fold_4['gender'] == 'f'))]
        
        self.train_image_paths, self.train_image_genders, self.train_classes = [], [], []
        self.genders = np.arange(2) # (male, female)
        self.gender_id_to_label = {1: 'female', 0: 'male'} # mapping from gender id (0/1) to its string label
        self.gender_label_to_id = {v:k for k,v in self.gender_id_to_label.items()}
        for fold in [fold_0, fold_1, fold_2, fold_3]:
            for user_id, image_name, face_id, gender, age in zip(fold['user_id'], fold['original_image'], 
                                                     fold['face_id'], fold['gender'], fold['age']):
                self.train_image_paths.append(f'{base_dir}/aligned/{user_id}/landmark_aligned_face.{face_id}.{image_name}')
                self.train_image_genders.append(1 if gender == 'f' else 0) # 1 for females and 0 for males
                self.train_classes.append(self.resolve_class_label(age))
        
        self.test_image_paths, self.test_image_genders, self.test_classes = [], [], []
        for user_id, image_name, face_id, gender, age in zip(fold_4['user_id'], fold_4['original_image'], 
                                                     fold_4['face_id'], fold_4['gender'], fold_4['age']):
                self.test_image_paths.append(f'{base_dir}/aligned/{user_id}/landmark_aligned_face.{face_id}.{image_name}')
                self.test_image_genders.append(1 if gender == 'f' else 0) # 1 for females and 0 for males
                self.test_classes.append(self.resolve_class_label(age))
        
#         self.loaded_images_test = [self.load_image(pth) for pth in self.test_image_paths]
#         self.loaded_images_train = [self.load_image(pth) for pth in self.train_image_paths]
    
    def length(self, portion):
        if portion == 'train':
            return len(self.train_image_paths)
        elif portion == 'test':
            return len(self.test_image_paths)
        else:
            raise ValueError("Portion {} not understood".format(portion))
    
    def num_classes(self):
        return len(self.classes)
    
    def load_image(self, image_path):
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = self.data_transform(image)
        return image
    
    def get_image(self, portion, idx):
        if portion == 'train':
            return self.load_image(self.train_image_paths[idx])
#             return self.loaded_images_train[idx]
        elif portion == 'test':
            return self.load_image(self.test_image_paths[idx])
#             return self.loaded_images_test[idx]
        else:
            raise ValueError("Portion {} not understood".format(portion))
    
    def get_image_protected_id_to_label(self, protected_id, attr='gender'):
        if attr not in ['gender']:
            raise ValueError('{} not an acceptable attr. Must be one of {}'.format(attr, ['gender']))

        return self.gender_id_to_label[protected_id]
    
    def get_image_protected_label_to_id(self, protected_label, attr='gender'):
        if attr not in ['gender']:
            raise ValueError('{} not an acceptable attr. Must be one of {}'.format(attr, ['gender']))
        
        return self.gender_label_to_id[protected_label]

    
    def get_image_label(self, portion, idx):
        if portion == 'train':
            return self.train_classes[idx]
        elif portion == 'test':
            return self.test_classes[idx]
        else:
            raise ValueError("Portion {} not understood".format(portion))
    
    def get_image_protected_class(self, portion, idx):
        if portion == 'train':
            return self.train_image_genders[idx]
        elif portion == 'test':
            return self.test_image_genders[idx]
        else:
            raise ValueError("Portion {} not understood".format(portion))
    
    def resolve_class_label(self, age):
        if age == '(0, 2)' or age == '2':
            age_id = 0
        elif age == '(4, 6)' or age == '3':
            age_id = 1
        elif age == '(8, 12)' or age == '(8, 23)' or age == '13':
            age_id = 2
        elif age == '(15, 20)' or age == '22':
            age_id = 3
        elif age == '(25, 32)' or age == '(27, 32)' or age in ['23', '29', '34', '35']:
            age_id = 4
        elif age == '(38, 42)' or age == '(38, 43)' or age == '(38, 48)' or age in ['36', '42', '45']:
            age_id = 5
        elif age == '(48, 53)' or age in ['46', '55']:
            age_id = 6
        elif age == '(60, 100)' or age in ['57', '58']:
            age_id = 7
        else:
            raise ValueError("Not sure how to handle this age: {}".format(age))
        
        return age_id

# In[5]:


def local_experiments_Adience():
    """
    Putting it in a function so that it doesn't execute when imported from another file
    """
    fold_0 = pd.read_csv('../data/adience/fold_0_data.txt', sep='\t')
    fold_1 = pd.read_csv('../data/adience/fold_1_data.txt', sep='\t')
    fold_2 = pd.read_csv('../data/adience/fold_2_data.txt', sep='\t')
    fold_3 = pd.read_csv('../data/adience/fold_3_data.txt', sep='\t')
    fold_4 = pd.read_csv('../data/adience/fold_4_data.txt', sep='\t')
    
    print ('Total entries: {}'.format(len(fold_0) + len(fold_1) + len(fold_2) + len(fold_3) + len(fold_4)))
    
    print ('Total Images: {}'.format(len(glob.glob('../data/adience/aligned/*/*'))))
    
    fold_0 = fold_0[np.logical_and(fold_0['age'] != 'None', 
                                   np.logical_or(fold_0['gender'] == 'm', fold_0['gender'] == 'f')
                                  )]
    fold_1 = fold_1[np.logical_and(fold_1['age'] != 'None', 
                                   np.logical_or(fold_1['gender'] == 'm', fold_1['gender'] == 'f')
                                  )]
    fold_2 = fold_2[np.logical_and(fold_2['age'] != 'None', 
                                   np.logical_or(fold_2['gender'] == 'm', fold_2['gender'] == 'f')
                                  )]
    fold_3 = fold_3[np.logical_and(fold_3['age'] != 'None', 
                                   np.logical_or(fold_3['gender'] == 'm', fold_3['gender'] == 'f')
                                  )]
    fold_4 = fold_4[np.logical_and(fold_4['age'] != 'None', 
                                   np.logical_or(fold_4['gender'] == 'm', fold_4['gender'] == 'f')
                                  )]
    print ('Total entries with age and gender: {}'.format(
        len(fold_0) + len(fold_1) + len(fold_2) + len(fold_3) + len(fold_4)))
    print (sorted(set(fold_4['age']).union(set(fold_3['age'])).union(set(fold_2['age'])).union(
        set(fold_1['age'])).union(set(fold_0['age']))))
    
    import os
    for user_id, image_name, face_id, gender, age in zip(fold_3['user_id'], fold_3['original_image'], 
                                                         fold_3['face_id'], fold_3['gender'], fold_3['age']):
    #     print ('../data/adience/aligned/{}/{}'.format(user_id, image_name))
    #     print (glob.glob('../data/adience/aligned/{}/landmark_aligned_face.{}.{}'.format(
    #         user_id, face_id, image_name)))
#         print (gender == 'f', age)
        assert len(glob.glob('../data/adience/aligned/{}/landmark_aligned_face.{}.{}'.format(
            user_id, face_id, image_name))) == 1
    
    import imageio
    arr1 = imageio.imread(
        '../data/adience/aligned/114841417@N06/landmark_aligned_face.483.12085238366_58ba30f728_o.jpg')
    arr2 = imageio.imread(
        '../data/adience/aligned/114841417@N06/landmark_aligned_face.490.12085238366_58ba30f728_o.jpg')
    
    from matplotlib import pyplot as plt
    print (arr2.shape)
    plt.imshow(arr1)
    plt.show()
    
    plt.imshow(arr2)
    plt.show()


# In[6]:


class UTKFace(Dataloader):
    
    """
    Class that does the dirty loading for the UTKFace dataset
    """
    
    def __init__(self, name, root_dir='.', load_filenames=True):
        #TODO: allow for specifying protected classes and target class (via optional parameters)
        #TODO: allow for different distributions of protected classes (via optional parameters)
        random.seed(42)

        directory = '{}/../data/UTKFace/'.format(root_dir) # directory with the images
        
        self.name = name
        self.data_transform = transforms.Compose([ # place any needed transforms here
                transforms.ToTensor(),
                transforms.Normalize([0., 0., 0.], [1., 1., 1.])
            ])
        
        # properties of the images
        self.ages = np.arange(1,117) # age in years
        self.classes = ['0-15', '15-25', '25-40', '40-60', '60+']  # bins of ages (see resolve_class_label())
        self.genders = np.arange(2) # (male, female)
        self.gender_id_to_label = {0: 'male', 1: 'female'} # mapping from gender id (0/1) to its string label
        self.gender_label_to_id = {v:k for k,v in self.gender_id_to_label.items()}
        self.races = np.arange(5)  # (White, Black, Asian, Indian, Others)
        self.race_id_to_label = {0: 'white', 1: 'black', 2: 'asian', 3: 'indian', 4: 'other'}
        self.race_label_to_id = {v:k for k,v in self.race_id_to_label.items()}
#         self.race_id_to_label = {0: 'other', 1: 'black'}
        
        # Extract info from each image
        self.image_paths, self.image_ages, self.image_genders, self.image_races = [], [], [], []
        self.image_classes = []
        
        if load_filenames:
            self.load_paths(directory)
    
    def visual_samples(self):
        print ("First 5 train samples")
        for i in range(5):
            train_image, train_image_class, train_image_gender, train_image_race =                 self.get_image('train', int(i)), self.classes[self.get_image_label('train', int(i))],                 self.gender_id_to_label[self.get_image_protected_class('train', int(i), attr='gender')],                self.race_id_to_label[self.get_image_protected_class('train', int(i), attr='race')]
            train_image = np.moveaxis(train_image.numpy(), 0, -1)
            train_image = (train_image * self.data_transform.transforms[-1].std) +                 self.data_transform.transforms[-1].mean
            plot_image(train_image, save_fig=False,
                       plot_title='Class: {}, Gender: {}, Race: {}'.format(train_image_class,
                                                                                       train_image_gender,
                                                                                       train_image_race))
        print ("First 5 test samples")
        for i in range(5):
            test_image, test_image_class, test_image_gender, test_image_race =                 self.get_image('test', int(i)), self.classes[self.get_image_label('test', int(i))],                 self.gender_id_to_label[self.get_image_protected_class('test', int(i), attr='gender')],                self.race_id_to_label[self.get_image_protected_class('test', int(i), attr='race')]
            test_image = np.moveaxis(test_image.numpy(), 0, -1)
            test_image = (test_image * self.data_transform.transforms[-1].std) +                 self.data_transform.transforms[-1].mean
            plot_image(test_image, save_fig=False,
                       plot_title='Class: {}, Gender: {}, Race: {}'.format(test_image_class,
                                                                                       test_image_gender,
                                                                                       test_image_race))
    
    def load_paths(self, directory):
        filepath = os.fsencode(directory)
        # sort the o/p of listdir to ensure that the ordering is same always
        for file in sorted(os.listdir(filepath)):
            filename = os.fsdecode(file)
            try:
                self.image_paths = np.append(self.image_paths, directory+filename)
                age, gender, race, _ = filename.split('_')
                age, gender, race = int(age), int(gender), int(race)
                age_bin = self.resolve_class_label(age)
            except:
                pass
#                print('Error: Age, Gender, and/or Race Unknown')
               
#                img = imageio.imread(directory+filename)
#                plt.imshow(img)
#                plt.show()
               
#                print(filename)
               
#                manual_classification = input('Would you like to classify this image manually? (y/n)\n')
#                if manual_classification == 'y':
#                    age = input('Age (years old): \n')
#                    gender = input('Gender {0: male, 1: female}: \n')
#                    race = input('Race {0: white, 1: black, 2: asian, 3: indian, 4: other}: \n')
            
            self.image_classes = np.append(self.image_classes, int(age_bin))
            self.image_ages = np.append(self.image_ages, age)
            self.image_genders = np.append(self.image_genders, int(gender))
#             self.image_races = np.append(self.image_races, int(race == 1))
            self.image_races = np.append(self.image_races, int(race))
            
        # Split the data into train and test sets
        all_indices = list(range(len(self.image_paths)))
        random.shuffle(all_indices)
        
        train_cutoff = int(0.8 * len(all_indices)) # 80:20 train test split
        self.train_indices = all_indices[:train_cutoff]
        self.train_image_paths = self.image_paths[self.train_indices]
        self.train_image_classes = self.image_classes[self.train_indices].astype('int')
        self.train_image_ages = self.image_ages[self.train_indices].astype('int')
        self.train_image_genders = self.image_genders[self.train_indices].astype('int')
        self.train_image_races = self.image_races[self.train_indices].astype('int')
        
        self.test_indices = all_indices[train_cutoff:]
        self.test_image_paths = self.image_paths[self.test_indices]
        self.test_image_classes = self.image_classes[self.test_indices].astype('int')
        self.test_image_ages = self.image_ages[self.test_indices].astype('int')
        self.test_image_genders = self.image_genders[self.test_indices].astype('int')
        self.test_image_races = self.image_races[self.test_indices].astype('int')
        
        assert len(self.train_indices) + len(self.test_indices) == len(all_indices)

    def length(self, portion):
        if portion == 'train':
            return len(self.train_image_paths)
        elif portion == 'test':
            return len(self.test_image_paths)
        else:
            raise ValueError("Portion {} not understood".format(portion))
    
    def num_classes(self):
        return len(self.classes)
    
    def load_image(self, image_path):
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = self.data_transform(image)
        return image
    
    def get_image(self, portion, idx):
        if portion == 'train':
            return self.load_image(self.train_image_paths[idx])
#             return self.loaded_images_train[idx]
        elif portion == 'test':
            return self.load_image(self.test_image_paths[idx])
#             return self.loaded_images_test[idx]
        else:
            raise ValueError("Portion {} not understood".format(portion))
    
    def get_image_label(self, portion, idx):
        if portion == 'train':
            return self.train_image_classes[idx]
        elif portion == 'test':
            return self.test_image_classes[idx]
        else:
            raise ValueError("Portion {} not understood".format(portion))
    
    def get_image_protected_id_to_label(self, protected_id, attr='gender'):
        if attr not in ['gender', 'race']:
            raise ValueError('{} not an acceptable attr. Must be one of {}'.format(attr, ['gender', 'race']))
        
        return self.race_id_to_label[protected_id] if attr == 'race' else self.gender_id_to_label[protected_id]
    
    def get_image_protected_label_to_id(self, protected_label, attr='gender'):
        if attr not in ['gender', 'race']:
            raise ValueError('{} not an acceptable attr. Must be one of {}'.format(attr, ['gender', 'race']))
        
        return self.race_label_to_id[protected_label] if attr == 'race' else self.gender_label_to_id[protected_label]
    
    def get_image_protected_class(self, portion, idx, attr='gender'):
        ## gender is binary, however for race we consider black as the minority and every other race as majority
        if attr not in ['gender', 'race']:
            raise ValueError('{} not an acceptable attr. Must be one of {}'.format(attr, ['gender', 'race']))

        if portion == 'train':
            return self.train_image_genders[idx] if attr == 'gender' else self.train_image_races[idx]
        elif portion == 'test':
            return self.test_image_genders[idx] if attr == 'gender' else self.test_image_races[idx]
        else:
            raise ValueError("Portion {} not understood".format(portion))
    
    def resolve_class_label(self, age):
        if age in range(15):
            age_id = 0
        elif age in range(15,25):
            age_id = 1
        elif age in range(25,40):
            age_id = 2
        elif age in range(40,60):
            age_id = 3
        elif age >= 60:
            age_id = 4
        else:
            raise ValueError("Not sure how to handle this age: {}".format(age))
        
        return age_id
    

# In[7]:


def local_testing_UTKFace():
    obj = UTKFace(name='utkface')
    obj.visual_samples()


# In[8]:


# local_testing_UTKFace()


# In[18]:


class CIFAR10(Dataloader):
    """
    Does Dirty loading for CIFAR-10 dataset
    """
    
    
    def __init__(self, name, root_dir='.'):        
        """
        self.name: dataloader instance name
        self.classes: list with string names of classes. i.e. k-th element is name of class indicated by k.
        self.train_data: Each element is an image which is in turn a list of 3072 elements. The first, second 
                         and third 1024 elements are the red, green and blue values of the image, respectively.
        self.train_labels: labels for train_data
        self.test_data: same as train_data but for test data.
        self.test_labels: same as train_labels but contains test_data's labels.
        """
        self.name = name
        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        cifar10_train = torchvision.datasets.CIFAR100(root='{}/../data'.format(root_dir), train=True, 
                                                      download=True, transform=self.data_transform)
        cifar10_test = torchvision.datasets.CIFAR100(root='{}/../data'.format(root_dir), train=False, 
                                                      download=True, transform=self.data_transform)

        # Contains meta data including label_names
        meta = cifar10_train.meta
        # Encodes index to text representation of class. classes[k] gives the class name of the k-th class.
        self.classes = cifar10_train.classes
        
        self.train_data, self.train_labels = cifar10_train.data, cifar10_train.targets
        self.test_data, self.test_labels = cifar10_test.data, cifar10_test.targets
            

    # Returns number of points in portion
    def length(self, portion):
        if portion == Dataloader.TRAIN:
            return len(self.train_data)
        elif portion == Dataloader.TEST:
            return len(self.test_data)
        else:
            raise ValueError('Portion {} not understood.'.format(portion))

    def num_classes(self):
        return len(self.classes)
        
    # Returns flattened version of idx-th image in 'portion' split of dataset. Each image is array of length 1024*3
    # The first, second, and third groups of 1024 numbers are the red, green, and blue values respectively.
    def get_image(self, portion, idx):
        if portion == Dataloader.TRAIN:
            return self.data_transform(self.plottable(self.train_data[idx]))
        elif portion == Dataloader.TEST:
            return self.data_transform(self.plottable(self.test_data[idx]))
        else:
            raise ValueError('Portion {} not understood.'.format(portion))
            
    # Gets the integer image label
    def get_image_label(self, portion, idx):
        if portion == Dataloader.TRAIN:
            return self.train_labels[idx]
        elif portion == Dataloader.TEST:
            return self.test_labels[idx]
        else:
            raise ValueError('Portion {} not understood.'.format(portion))
        
    # Returns numpy array in format plottable by matplotlib.pyplot.imshow()
    # Precondition: parameter img to be in format given by get_image_label()
    def plottable(self, img):
        # Check dimensions of img
        if len(list(img)) != 1024*3:
            raise ValueError('Image size passed is not 3072.')
            
        red = img[0:1024].reshape((32,32))
        green = img[1024:2048].reshape((32,32))
        blue = img[2048:3072].reshape((32,32))
        plottable = np.array([red, green, blue])
        plottable = np.transpose(plottable, (1,2,0))
        
        return plottable
    
    # Create imbalance in data by cutting size of a certain class.
    # This actually edits the value of self.classes and cannot be undone.
    # To recover the value of 
    def cut_class(self, portion, class_idx, cut_percent):
        """
        The current implementation is not very efficient but that should be no problem since this function
        is only being called ~11 times.
        
        portion: TRAIN or TEST
        class_idx: index of class. See self.classes for corresponding classes
        cut_percent: Will cut selected class by cut_percent %. New size of class will be original_sz * (1-cut_percent)
        """
        if class_idx >= self.num_classes():
            raise ValueError('class_idx {} beyound num_classes:{}'.format(class_idx, self.num_classes()))
        
        if (cut_percent > 1) or (cut_percent < 0):
            raise ValueError('cut_percent {} must be between 0 and 1.'.format(cut_percent))
            
        if portion == Dataloader.TRAIN:       
            # size of class to be cut
            cut_class_sz = int(np.sum([1 for l in self.train_labels if l == class_idx]))
            # indeces to remove from dataset
            cut_idxs = [idx for idx,label in enumerate(self.train_labels) if label == class_idx]
            num_to_cut = round(cut_class_sz  * cut_percent) # number of labels of cut class we wish to cut out
            np.random.shuffle(cut_idxs) # shuffle indeces to cut to make selection random
            cut_idxs = cut_idxs[:num_to_cut] # resize cut_idxs

            # Take out from train data and labels the elements with indexes in cut_idxs
            self.train_data = [data for idx, data in enumerate(self.train_data) if idx not in cut_idxs]
            self.train_labels = [data for idx, data in enumerate(self.train_labels) if idx not in cut_idxs]
            
        elif portion == Dataloader.TEST:
            # size of class to be cut
            cut_class_sz = int(np.sum([1 for l in self.test_labels if l == class_idx]))
            # indeces to remove from dataset
            cut_idxs = [idx for idx,label in enumerate(self.test_labels) if label == class_idx]
            num_to_cut = round(cut_class_sz  * cut_percent) # number of labels of cut class we wish to cut out
            np.random.shuffle(cut_idxs) # shuffle indeces to cut to make selection random
            cut_idxs = cut_idxs[:num_to_cut] # resize cut_idxs

            # Take out from train data and labels the elements with indexes in cut_idxs
            self.test_data = [data for idx, data in enumerate(self.test_data) if idx not in cut_idxs]
            self.test_labels = [data for idx, data in enumerate(self.test_labels) if idx not in cut_idxs]
        else:
            raise ValueError('Portion {} not understood.'.format(portion))
        
    
    def class_size(self, portion, class_idx):
        """
        Returns the size of the class indexed by class_idx. Useful for debugging cut_class().
        """ 
        TEST_FULL = 10000  # full sizes for train and test splits
        TRAIN_FULL = 50000
        
        if portion == Dataloader.TRAIN:
            target_size = np.sum([1 for label in self.train_labels if label == class_idx])
            return target_size
            
        elif portion == Dataloader.TEST:
            target_size = np.sum([1 for label in self.test_labels if label == class_idx])
            return target_size
        
        else:
            raise ValueError('Portion {} not understood'.format(portion))
    
    # Helper function to load data batches.
    # Returns dictionary of data
    def __unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

class CIFAR100(Dataloader):
    """
    Does Dirty loading for CIFAR-10 dataset
    """
    
    
    def __init__(self, name, resize_for_alexnet=False, root_dir='.'):        
        """
        self.name: dataloader instance name
        self.classes: list with string names of classes. i.e. k-th element is name of class indicated by k.
        self.train_data: Each element is an image which is in turn a list of 3072 elements. The first, second 
                         and third 1024 elements are the red, green and blue values of the image, respectively.
        self.train_labels: labels for train_data
        self.test_data: same as train_data but for test data.
        self.test_labels: same as train_labels but contains test_data's labels.
        """
        self.name = name
        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), 
                                 (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))])
        self.data_transform_train = transforms.Compose([
            #transforms.ToPILImage(),
            #transforms.RandomCrop(32, padding=4),
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), 
                                 (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
        ])
        cifar100_training = torchvision.datasets.CIFAR100(root='{}/../data'.format(root_dir), train=True, 
                                                          download=True, transform=self.data_transform)
            
        self.data_transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), 
                                 (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
        ])
        #cifar100_test = CIFAR100Test(path, transform=transform_test)
        cifar100_test = torchvision.datasets.CIFAR100(root='{}/../data'.format(root_dir), train=False, 
                                                      download=True, transform=self.data_transform)

        if resize_for_alexnet:
            self.data_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), 
                                     (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))])

        
        
        
        # Contains meta data including label_names
        meta = cifar100_training.meta
        # Encodes index to text representation of class. classes[k] gives the class name of the k-th class.
        self.classes = cifar100_training.classes
        
        self.train_data, self.train_labels = cifar100_training.data, cifar100_training.targets
        self.test_data, self.test_labels = cifar100_test.data, cifar100_test.targets
                    

    # Returns number of points in portion
    def length(self, portion):
        if portion == Dataloader.TRAIN:
            return len(self.train_data)
        elif portion == Dataloader.TEST:
            return len(self.test_data)
        else:
            raise ValueError('Portion {} not understood.'.format(portion))

    def num_classes(self):
        return len(self.classes)
        
    # Returns flattened version of idx-th image in 'portion' split of dataset. Each image is array of length 1024*3
    # The first, second, and third groups of 1024 numbers are the red, green, and blue values respectively.
    def get_image(self, portion, idx):
        if portion == Dataloader.TRAIN:
            return self.data_transform(self.train_data[idx])
        elif portion == Dataloader.TEST:
            return self.data_transform(self.test_data[idx])
        else:
            raise ValueError('Portion {} not understood.'.format(portion))
            
    # Gets the integer image label
    def get_image_label(self, portion, idx):
        if portion == Dataloader.TRAIN:
            return self.train_labels[idx]
        elif portion == Dataloader.TEST:
            return self.test_labels[idx]
        else:
            raise ValueError('Portion {} not understood.'.format(portion))
        
    
    # Create imbalance in data by cutting size of a certain class.
    # This actually edits the value of self.classes and cannot be undone.
    # To recover the value of 
    def cut_class(self, portion, class_idx, cut_percent):
        """
        The current implementation is not very efficient but that should be no problem since this function
        is only being called ~11 times.
        
        portion: TRAIN or TEST
        class_idx: index of class. See self.classes for corresponding classes
        cut_percent: Will cut selected class by cut_percent %. New size of class will be original_sz * (1-cut_percent)
        """
        if class_idx >= self.num_classes():
            raise ValueError('class_idx {} beyound num_classes:{}'.format(class_idx, self.num_classes()))
        
        if (cut_percent > 1) or (cut_percent < 0):
            raise ValueError('cut_percent {} must be between 0 and 1.'.format(cut_percent))
            
        if portion == Dataloader.TRAIN:       
            # size of class to be cut
            cut_class_sz = int(np.sum([1 for l in self.train_labels if l == class_idx]))
            # indeces to remove from dataset
            cut_idxs = [idx for idx,label in enumerate(self.train_labels) if label == class_idx]
            num_to_cut = round(cut_class_sz  * cut_percent) # number of labels of cut class we wish to cut out
            np.random.shuffle(cut_idxs) # shuffle indeces to cut to make selection random
            cut_idxs = cut_idxs[:num_to_cut] # resize cut_idxs

            # Take out from train data and labels the elements with indexes in cut_idxs
            self.train_data = [data for idx, data in enumerate(self.train_data) if idx not in cut_idxs]
            self.train_labels = [data for idx, data in enumerate(self.train_labels) if idx not in cut_idxs]
            
        elif portion == Dataloader.TEST:
            # size of class to be cut
            cut_class_sz = int(np.sum([1 for l in self.test_labels if l == class_idx]))
            # indeces to remove from dataset
            cut_idxs = [idx for idx,label in enumerate(self.test_labels) if label == class_idx]
            num_to_cut = round(cut_class_sz  * cut_percent) # number of labels of cut class we wish to cut out
            np.random.shuffle(cut_idxs) # shuffle indeces to cut to make selection random
            cut_idxs = cut_idxs[:num_to_cut] # resize cut_idxs

            # Take out from train data and labels the elements with indexes in cut_idxs
            self.test_data = [data for idx, data in enumerate(self.test_data) if idx not in cut_idxs]
            self.test_labels = [data for idx, data in enumerate(self.test_labels) if idx not in cut_idxs]
        else:
            raise ValueError('Portion {} not understood.'.format(portion))
        
    
    def class_size(self, portion, class_idx):
        """
        Returns the size of the class indexed by class_idx. Useful for debugging cut_class().
        """ 
        TEST_FULL = 10000  # full sizes for train and test splits
        TRAIN_FULL = 50000
        
        if portion == Dataloader.TRAIN:
            target_size = np.sum([1 for label in self.train_labels if label == class_idx])
            return target_size
            
        elif portion == Dataloader.TEST:
            target_size = np.sum([1 for label in self.test_labels if label == class_idx])
            return target_size
        
        else:
            raise ValueError('Portion {} not understood'.format(portion))


CIFAR100_LABELS_LIST = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
    'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank',
    'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
    'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]
CIFAR100_SUPERCLASS_LABELS_LIST = [
    'aquatic_mammals', 'fish', 'flowers', 'food_containers',
    'fruit_and_vegetables', 'household_electrical_devices',
    'household_furniture', 'insects', 'large_carnivores',
    'large_man-made_outdoor_things', 'large_natural_outdoor_scenes',
    'large_omnivores_and_herbivores', 'medium_mammals',
    'non-insect_invertebrates', 'people', 'reptiles', 'small_mammals',
    'trees', 'vehicles_1', 'vehicles_2'
]
CIFAR100_CLASSES_LABELS_LIST = [
    'beaver', 'dolphin', 'otter', 'seal', 'whale',
    'aquarium_fish', 'flatfish', 'ray', 'shark', 'trout',
    'orchid', 'poppy', 'rose', 'sunflower', 'tulip',
    'bottle', 'bowl', 'can', 'cup', 'plate',
    'apple', 'mushroom', 'orange', 'pear', 'sweet_pepper',
    'clock', 'keyboard', 'lamp', 'telephone', 'television',
    'bed', 'chair', 'couch', 'table', 'wardrobe',
    'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',
    'bear', 'leopard', 'lion', 'tiger', 'wolf',
    'bridge', 'castle', 'house', 'road', 'skyscraper',
    'cloud', 'forest', 'mountain', 'plain', 'sea',
    'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
    'fox', 'porcupine', 'possum', 'raccoon', 'skunk',
    'crab', 'lobster', 'snail', 'spider', 'worm',
    'baby', 'boy', 'girl', 'man', 'woman',
    'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',
    'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',
    'maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree',
    'bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train',
    'lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor'
]


class CIFAR100super(Dataloader):
    """
    Does Dirty loading for CIFAR-10 dataset
    """
    
    
    def __init__(self, name, resize_for_alexnet=False, root_dir='.'):        
        """
        self.name: dataloader instance name
        self.classes: list with string names of classes. i.e. k-th element is name of class indicated by k.
        self.train_data: Each element is an image which is in turn a list of 3072 elements. The first, second 
                         and third 1024 elements are the red, green and blue values of the image, respectively.
        self.train_labels: labels for train_data
        self.test_data: same as train_data but for test data.
        self.test_labels: same as train_labels but contains test_data's labels.
        """
        self.name = name
        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), 
                                 (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))])
        self.data_transform_train = transforms.Compose([
            #transforms.ToPILImage(),
            #transforms.RandomCrop(32, padding=4),
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), 
                                 (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
        ])
        cifar100_training = torchvision.datasets.CIFAR100(root='{}/../data'.format(root_dir), train=True, 
                                                          download=True, transform=self.data_transform)
            
        self.data_transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), 
                                 (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
        ])
        #cifar100_test = CIFAR100Test(path, transform=transform_test)
        cifar100_test = torchvision.datasets.CIFAR100(root='{}/../data'.format(root_dir), train=False, 
                                                      download=True, transform=self.data_transform)

        if resize_for_alexnet:
            self.data_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), 
                                     (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))])
        
        
        # Contains meta data including label_names
        meta = cifar100_training.meta
        # Encodes index to text representation of class. classes[k] gives the class name of the k-th class.
        self.classes = CIFAR100_SUPERCLASS_LABELS_LIST
        
        cifar100super_train_lab = [math.floor(CIFAR100_CLASSES_LABELS_LIST.index(CIFAR100_LABELS_LIST[x]) / 5) \
                                   for x in cifar100_training.targets]
        cifar100super_test_lab = [math.floor(CIFAR100_CLASSES_LABELS_LIST.index(CIFAR100_LABELS_LIST[x]) / 5) \
                                  for x in cifar100_test.targets]

        
        self.train_data, self.train_labels = cifar100_training.data, cifar100super_train_lab
        self.test_data, self.test_labels = cifar100_test.data, cifar100super_test_lab
                    

    # Returns number of points in portion
    def length(self, portion):
        if portion == Dataloader.TRAIN:
            return len(self.train_data)
        elif portion == Dataloader.TEST:
            return len(self.test_data)
        else:
            raise ValueError('Portion {} not understood.'.format(portion))

    def num_classes(self):
        return len(self.classes)
        
    # Returns flattened version of idx-th image in 'portion' split of dataset. Each image is array of length 1024*3
    # The first, second, and third groups of 1024 numbers are the red, green, and blue values respectively.
    def get_image(self, portion, idx):
        if portion == Dataloader.TRAIN:
            return self.data_transform(self.train_data[idx])
        elif portion == Dataloader.TEST:
            return self.data_transform(self.test_data[idx])
        else:
            raise ValueError('Portion {} not understood.'.format(portion))
            
    # Gets the integer image label
    def get_image_label(self, portion, idx):
        if portion == Dataloader.TRAIN:
            return self.train_labels[idx]
        elif portion == Dataloader.TEST:
            return self.test_labels[idx]
        else:
            raise ValueError('Portion {} not understood.'.format(portion))
        
    
    # Create imbalance in data by cutting size of a certain class.
    # This actually edits the value of self.classes and cannot be undone.
    # To recover the value of 
    def cut_class(self, portion, class_idx, cut_percent):
        """
        The current implementation is not very efficient but that should be no problem since this function
        is only being called ~11 times.
        
        portion: TRAIN or TEST
        class_idx: index of class. See self.classes for corresponding classes
        cut_percent: Will cut selected class by cut_percent %. New size of class will be original_sz * (1-cut_percent)
        """
        if class_idx >= self.num_classes():
            raise ValueError('class_idx {} beyound num_classes:{}'.format(class_idx, self.num_classes()))
        
        if (cut_percent > 1) or (cut_percent < 0):
            raise ValueError('cut_percent {} must be between 0 and 1.'.format(cut_percent))
            
        if portion == Dataloader.TRAIN:       
            # size of class to be cut
            cut_class_sz = int(np.sum([1 for l in self.train_labels if l == class_idx]))
            # indeces to remove from dataset
            cut_idxs = [idx for idx,label in enumerate(self.train_labels) if label == class_idx]
            num_to_cut = round(cut_class_sz  * cut_percent) # number of labels of cut class we wish to cut out
            np.random.shuffle(cut_idxs) # shuffle indeces to cut to make selection random
            cut_idxs = cut_idxs[:num_to_cut] # resize cut_idxs

            # Take out from train data and labels the elements with indexes in cut_idxs
            self.train_data = [data for idx, data in enumerate(self.train_data) if idx not in cut_idxs]
            self.train_labels = [data for idx, data in enumerate(self.train_labels) if idx not in cut_idxs]
            
        elif portion == Dataloader.TEST:
            # size of class to be cut
            cut_class_sz = int(np.sum([1 for l in self.test_labels if l == class_idx]))
            # indeces to remove from dataset
            cut_idxs = [idx for idx,label in enumerate(self.test_labels) if label == class_idx]
            num_to_cut = round(cut_class_sz  * cut_percent) # number of labels of cut class we wish to cut out
            np.random.shuffle(cut_idxs) # shuffle indeces to cut to make selection random
            cut_idxs = cut_idxs[:num_to_cut] # resize cut_idxs

            # Take out from train data and labels the elements with indexes in cut_idxs
            self.test_data = [data for idx, data in enumerate(self.test_data) if idx not in cut_idxs]
            self.test_labels = [data for idx, data in enumerate(self.test_labels) if idx not in cut_idxs]
        else:
            raise ValueError('Portion {} not understood.'.format(portion))
        
    
    def class_size(self, portion, class_idx):
        """
        Returns the size of the class indexed by class_idx. Useful for debugging cut_class().
        """ 
        TEST_FULL = 10000  # full sizes for train and test splits
        TRAIN_FULL = 50000
        
        if portion == Dataloader.TRAIN:
            target_size = np.sum([1 for label in self.train_labels if label == class_idx])
            return target_size
            
        elif portion == Dataloader.TEST:
            target_size = np.sum([1 for label in self.test_labels if label == class_idx])
            return target_size
        
        else:
            raise ValueError('Portion {} not understood'.format(portion))
