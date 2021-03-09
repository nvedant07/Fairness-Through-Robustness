#!/usr/bin/env python
# coding: utf-8

# ### Defines helper functions; inlcuding those for plotting

# In[1]:


import torch
import sys, os, glob
import numpy as np
import pickle, joblib
import seaborn as sns

from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

sys.path.insert(0, '../util')

from pytorch_data_loader import PytorchLoader
from data_loader import UTKFace, UTKFace_C, Adience, Adience_C, CIFAR10, CIFAR10_C, \
    CIFAR100, CIFAR100_C, CIFAR100super, CIFAR100super_C


# In[ ]:


DATASET_TO_OBJECT_MAPPING = {'cifar10': CIFAR10,
                             'utkface': UTKFace,
                             'adience': Adience,
                             'cifar100': CIFAR100,
                             'cifar100super': CIFAR100super}
DATASET_TO_MODEL_NAMES = {'cifar10': ['deep_cnn', 'mlp1_cifar', 'conv2_cifar',
                                      'resnet','alexnet',
                                      'vgg','squeezenet','densenet'], 
                          'utkface': ['utk_classifier', 'resnet','alexnet',
                                      'vgg','squeezenet','densenet'], 
                          'adience': ['adience_classifier', 'resnet','alexnet',
                                      'vgg','squeezenet','densenet']} 
#                           'adience': ['vgg16', 'vgg19']}

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']

# In[ ]:

def set_paper_friendly_plots_params():
    plt.style.use('seaborn-paper')
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 22
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titlesize'] = 15
    plt.rcParams['axes.linewidth'] = 1.25
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['legend.fontsize'] = 20
    plt.rcParams['figure.titlesize'] = 22
    plt.rcParams['lines.linewidth'] = 4.0
    plt.rcParams['grid.color'] = 'grey'
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.linewidth'] = 0.25
    plt.rcParams['figure.dpi'] = 50
    plt.rcParams['savefig.dpi'] = 50


def get_data_loder_objects(dataset_name, phases, root_dir='.', ds_kwargs={}, **kwargs):
    """
    kwargs: stuff like batch_size, shuffle, num_workers etc fed into PyTorch's DataLoader object
    ds_kwargs: stuff that goes into the ds_obj init functions. Only needed for *_C datasets
        {'distortion_type': str, 'severity': int}
    """
    stem_dataset_name = dataset_name.split('_')[0]
    ds_obj = DATASET_TO_OBJECT_MAPPING[stem_dataset_name.lower()](dataset_name, root_dir=root_dir, **ds_kwargs)
    # Wrap the DataLoader object via a PyTorchLoader
    datasets = {x : PytorchLoader(ds=ds_obj, portion=x) for x in phases}
    data_loaders = {"{}".format(x) : torch.utils.data.DataLoader(datasets[x], **kwargs) for x in phases}
    return ds_obj, datasets, data_loaders


# In[ ]:


def get_loader_kwargs(batch_size):
    return {'batch_size': batch_size, 'shuffle': True, 'num_workers': 4}


# In[ ]:


def get_model_names(dataset_name, with_regularization=False):
    if not with_regularization:
        return DATASET_TO_MODEL_NAMES[dataset_name.split('_')[0].lower()]
    else:
        return [x + '_regularized' for x in DATASET_TO_MODEL_NAMES[dataset_name.split('_')[0].lower()]]


# In[ ]:


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# In[ ]:


def persist_model_weights(model, ds, learning_rate, epoch, root_dir='.'):
    """
    model: instance of class DNN defined in model.ipynb
    ds: instance of DataLoader defined in data_loader.ipynb
    learning_rate: to be included in filename 
    (TODO: this isn't really needed since learning rate is an attr of model)
    epoch: refers to the weight of epochs that we're saving
    """
    create_dir("{}/../{}".format(root_dir, ds.name))
    create_dir("{}/../{}/model_weights".format(root_dir, ds.name))
    filename = "{}/../{}/model_weights/{}_{}_epoch_{}_lr_{}.pth".format(root_dir, ds.name, 
            model.model_name, model.criterion._get_name(), epoch, learning_rate)
    torch.save(model.model_ft.state_dict(), filename)
    print ("Model saved to: {}".format(filename))

def persist_epoch_values(model, ds, learning_rate, values, values_names, root_dir='.'):
    create_dir("{}/../{}".format(root_dir, ds.name))
    create_dir("{}/../{}/training_values".format(root_dir, ds.name))
    for val, val_name in zip(values, values_names):
        filename = "{}/../{}/training_values/{}_{}_lr_{}_{}.pkl".format(root_dir, ds.name, 
            model.model_name, model.criterion._get_name(), learning_rate, val_name)
        joblib.dump(val, filename)
        print ("Saved {} to {}!".format(val_name, filename))

def load_epoch_values(model, ds, learning_rate, values_names, root_dir='.'):
    loaded_vals = []
    for val_name in values_names:
        filename = "{}/../{}/training_values/{}_{}_lr_{}_{}.pkl".format(root_dir, ds.name, 
            model.model_name, model.criterion._get_name(), learning_rate, val_name)
        loaded_vals.append(joblib.load(filename))
    return loaded_vals

# In[ ]:


def inverse_transpose_images(image, data_transform):
    """
    image is a numpy array of shape (3, n, n) where image is of size n x n
    data_transform is the attribute of ds which specifies the mean and std to be applied during pre-processing
    
    this function a. converts image to channels last, and then reverses the transformation which 
    pytorch's data loader would've applied
    """
    if isinstance(image, torch.Tensor):
        if image.device.type == 'cpu':
            image = image.numpy()
        else:
            image = image.cpu().numpy()
    if len(image.shape) == 4: # this means multiple batches
        image = np.moveaxis(image, 1, -1) # image is a channels first image
    else:
        image = np.moveaxis(image, 0, -1)
    image = (image * data_transform.transforms[-1].std) + data_transform.transforms[-1].mean
    return image


# In[ ]:


from matplotlib import pyplot as plt

def line_plots(lines, x_vals, x_label, y_label, filename, title, subfolder='none', legend_vals=None):
    """
    Generic function to plot (multiple) lines
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, line in enumerate(lines):
        ax.plot(x_vals, line, color=COLORS[i], 
                label=legend_vals[i] if legend_vals is not None else '')
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    if legend_vals is not None:
        ax.legend(loc='best')
    
    create_dir('plots')
    create_dir('plots/{}'.format(subfolder))

    plt.savefig('plots/{}/{}'.format(subfolder, filename), bbox_inches='tight')
    plt.show()


# In[ ]:

def line_plots_grid(all_lines, all_x_vals, x_label, y_label, filename, titles, global_title, 
    subfolder, y_lims=(0,1), columns=2):
    """
    Plot multiple subplots, each of which will be a line plot
    """

    if paper_friendly_plot:
        set_paper_friendly_plots_params()
        extension = 'pdf'
    else:
        extension = 'png'

    columns = columns if len(all_lines) > columns else len(all_lines)
    rows = int(len(all_lines)/columns)
    fig = plt.figure(figsize=(5, 3))

    x_vals = np.arange(1, len(all_lines[0][0]) + 1)

    for i in range(1, columns * rows + 1):
        ax = fig.add_subplot(rows, columns, i)
        ax.set_ylim(y_lims)
        for j in range(len(all_lines[i-1])):
            ax.plot(all_x_vals[i - 1], all_lines[i-1][j], color=COLORS[j])
        # ax.set_xticks(x_vals)
        # ax.set_xticklabels(all_x_vals[i - 1])
        ax.set_title(titles[i-1])
    fig.suptitle(global_title)
    fig.savefig('{}/{}.{}'.format(subfolder, filename, extension), bbox_inches='tight')
    plt.close()



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
        plt.savefig('{}/{}/{}.{}'.format(root_folder, subfolder, filename, extension), bbox_inches='tight')
    if show_fig:
        plt.show()
    plt.clf()
    plt.close()

def stitched_images(image_objects, column_titles, results_dir, filename, extension, global_title='', columns=5, savefig=True, plot_title_colors=None, figsize=(15,10)):
    sns.set_style('white')
    if savefig:
        create_dir(results_dir)

    if plot_title_colors is not None:
        assert len(plot_title_colors) == len(column_titles)

    columns = columns if np.product(image_objects.shape[:2]) > columns else np.product(image_objects.shape[:2])
    rows = int(np.product(image_objects.shape[:2])/columns)
    fig = plt.figure(figsize=figsize)
    for i in range(1, columns * rows + 1):
        row_idx = int((i - 1) / columns)
        col_idx = int((i - 1) - row_idx * columns)
        ax = fig.add_subplot(rows, columns, i)
        ax.axis('off')
        ax.imshow(image_objects[row_idx][col_idx] if image_objects[row_idx][col_idx].shape[-1] == 3 else \
            image_objects[row_idx][col_idx].reshape((image_objects[row_idx][col_idx].shape[0], image_objects[row_idx][col_idx].shape[1])), 
            cmap='viridis' if image_objects[row_idx][col_idx].shape[-1] == 3 else 'gray',
            interpolation="bilinear")
        if i - 1 < len(column_titles):
            ax.set_title(column_titles[i-1], fontsize=10, color=plot_title_colors[i-1] if plot_title_colors is not None else 'black')
    fig.suptitle(global_title)
    if savefig:
        plt.savefig('{}/{}.{}'.format(results_dir, filename, extension), bbox_inches='tight')
        print ("Saved fig at {}/{}.{}".format(results_dir, filename, extension))
    plt.show()
    plt.close()
    return '{}/{}.{}'.format(results_dir, filename, extension)



# In[ ]:

class DistanceObject:

    def __init__(self, value):
        self.value = value

# class AdversarialObject:

#     def __init__(self, image, distance, total_prediction_calls, total_gradient_calls):
#         self.image = image
#         self.distance = distance
#         self._total_prediction_calls = total_prediction_calls
#         self._total_gradient_calls = total_gradient_calls

class AdversarialObject:

    def __init__(self, image, prediction, image_id):
        self.image = image
        self.prediction = prediction
        self.image_id = image_id




def make_pickleable(objects, predictions, image_ids):
    pickleable_objects = []
    for obj, pred, img_id in zip(objects, predictions, image_ids):
#         d = DistanceObject(obj.distance.value)
        adv = AdversarialObject(image=obj, prediction=pred, image_id=img_id)
        pickleable_objects.append(adv)
    return pickleable_objects


# In[ ]:



def save_objects(adversarial_objects, model_name, adversarial_image_ids, ds_obj, 
                 attack_obj, epoch, post_fix='', root_dir='.'):
    create_dir("{}/{}/".format(root_dir, ds_obj.name))
    create_dir("{}/{}/adversarial_images".format(root_dir, ds_obj.name))
    create_dir("{}/{}/adversarial_images/{}".format(root_dir, ds_obj.name, model_name))
    create_dir("{}/{}/adversarial_images/{}/{}".format(root_dir, ds_obj.name, model_name, attack_obj.name))
    for obj, image_id in zip(adversarial_objects, adversarial_image_ids):
        with open("{}/{}/adversarial_images/{}/{}/{}_epoch_{}{}.pkl".format(root_dir, ds_obj.name, model_name, 
                                            attack_obj.name, int(image_id), epoch, post_fix), 'wb') as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print ("Saved adversarial objects at: {}/{}/adversarial_images/{}/{}".format(root_dir, ds_obj.name, 
                                                                                 model_name, attack_obj.name))


# In[ ]:


def get_class_wise_accuracy(predicted_classes, true_classes, classes):
    predicted_classes, true_classes = np.array(predicted_classes), np.array(true_classes)
    class_to_accuracy = {}
    for class_id in range(len(classes)):
        mask = true_classes == class_id
        class_to_accuracy[classes[class_id]] = accuracy_score(true_classes[mask], predicted_classes[mask])
    return class_to_accuracy


# In[ ]:


def load_adversarial_objects(folder, epoch, ds_obj, device):
    object_paths = glob.glob("{}/*_epoch_{}*".format(folder, epoch))
    objects = []
    # predicted = []
    image_ids = []
    for obj in object_paths:
        image_ids.append(int(obj.split('/')[-1].rstrip('.pkl').split('_')[0]))
        adv_obj = joblib.load(obj)
        # tiled_mean = np.tile(
        #     ds_obj.data_transform.transforms[-1].mean, (adv_obj.image.shape[1], adv_obj.image.shape[2], 1)).T
        # tiled_std = np.tile(
        #     ds_obj.data_transform.transforms[-1].std, (adv_obj.image.shape[1], adv_obj.image.shape[2], 1)).T
        # processed_image = torch.tensor((adv_obj.image - tiled_mean)/tiled_std)
        # model_op = model.model_ft(processed_image.view((1,) + processed_image.size()).to(device))
        # _, preds = torch.max(model_op, 1)
        # predicted.append(preds.cpu().numpy()[0])
        objects.append(adv_obj)
    return np.array(image_ids), objects


# In[ ]:


def sigmoid(z):
    exp_fn = np.exp if isinstance(z, np.ndarray) else torch.exp    
    return 1 / (1 + exp_fn(-z))


# In[ ]:


def concatenate_tup_entries(tup_entries):
    """
    Given a list of tuples/lists, it concatenates all entries:
    Eg: [([1,2,3], [4,5,6], [True, False, False]), ([1,2,3], [4,5,6], [True, False, False])]
    will be returned as:
    [[1,2,3,1,2,3], [4,5,6,4,5,6], [True, False, False, True, False, False]]
    """
    if isinstance(tup_entries[0][0], torch.Tensor):
        cat_func = torch.cat
    elif isinstance(tup_entries[0][0], np.ndarray):
        cat_func = np.concatenate
    concatenated_tup = [None] * len(tup_entries[0])
    for entry in tup_entries:
        for idx, val in entry:
            concatenated_tup[idx] = val if concatenated_tup[idx] is None else                 cat_func((concatenated_tup[idx], val))
    return concatenated_tup


# In[ ]:


# src: https://stackoverflow.com/questions/45718523/pass-kwargs-to-starmap-while-using-pool-in-python
from itertools import repeat

def starmap_with_kwargs(pool, fn, kwargs_iter):
    args_for_starmap = zip(repeat(fn), kwargs_iter)
    return pool.starmap(apply_kwargs, args_for_starmap)

def apply_kwargs(fn, kwargs):
    return fn(**kwargs)


# In[ ]:


def prepare_kwargs(kwargs, inputs, protected_classes, phase):
    ## This takes in kwargs for regularized training and sets up the values for keys 
    ## 'inputs' and 'protected_classes'
    if 'inputs' in kwargs and 'protected_classes' in kwargs:
        kwargs.pop('inputs')
        kwargs['inputs'] = inputs
        kwargs.pop('protected_classes')
        kwargs['protected_classes'] = protected_classes
        kwargs['phase'] = phase
    return kwargs

