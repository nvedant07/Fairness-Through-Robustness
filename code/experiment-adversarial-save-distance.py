#### SAVES THE DISTANCES COMPUTED USING ADVERSARIAL SAMPLES TO A CSV FILE FOR EASY USE

import matplotlib as mpl
mpl.use('Agg')

import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score, classification_report

import sys, os, glob
import time
import operator
import itertools
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import foolbox
import getopt
sys.path.insert(0, "../util")


import model


## Load other helper functions and classes
from pytorch_data_loader import PytorchLoader
import helper as hp
from data_loader import UTKFace, Adience, CIFAR10
from adversarial import Attack, AttackV2


DATASET_TO_MODEL_NAMES = {
    'utkface': ['utk_classifier_regularized', 'resnet_regularized', 'vgg_regularized', 'densenet_regularized'], 
    'cifar10': ['deep_cnn_regularized', 'resnet_regularized', 'vgg_regularized', 'densenet_regularized']
    }

DATASET_TO_MODEL_TO_TAUS = {
    'utkface': {'utk_classifier_regularized': [2.0, 5.0], 'densenet_regularized': [2.0], 'resnet_regularized': [2.0], 'vgg_regularized': [2.0]},
    'cifar10': {'deep_cnn_regularized': [2.0], 'densenet_regularized': [2.0], 'resnet_regularized': [2.0], 'vgg_regularized': [2.0]}
}

DATASET_TO_MODEL_TO_ALPHAS = {
    'utkface': {'utk_classifier_regularized': [0.1, 1.0, 2.0, 10.0], 'densenet_regularized': [0.1, 1.0], 'resnet_regularized': [0.1, 1.0], 'vgg_regularized': [0.1, 1.0]},
    'cifar10': {'deep_cnn_regularized': [0.1, 1.0], 'densenet_regularized': [0.1, 1.0], 'resnet_regularized': [0.1, 1.0], 'vgg_regularized': [0.1, 1.0]}
}

DATASET_TO_MODEL_TO_BETAS = {
    
}

DATASET_TO_MODEL_TO_GAMMAS = {
    
}

with_regularization=True
sigmoid_approx=False
probabilities=True

PHASES = ['train', 'test']
batch_size = 500
learning_rate = 0.0005
aggregate_coeff = 5

paper_friendly_plots = True

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

def main(all_datasets, gpu, epoch):

    device = torch.device('cuda:{}'.format(gpu))

    attack_names = ['DeepFool', 'CarliniWagner']

    for attack_name in attack_names:
        csv_rows = []
        for dataset in all_datasets:
            ds_obj, datasets, data_loaders = \
                hp.get_data_loder_objects(dataset, PHASES, **hp.get_loader_kwargs(batch_size))

            for dir_name in os.listdir('../{}/adversarial_images/'.format(ds_obj.name)):
                # dir_name contains model name and other params, process them here
                if 'RegularizedLoss' in dir_name:
                    model_name = dir_name.split('_RegularizedLoss_')[0]
                    with_regularization = True
                    tau = float(dir_name.split('_tau_')[1].split('_')[0])
                    alpha = float(dir_name.split('_alpha_')[1].split('_')[0])
                    if 'probabilities' in dir_name:
                        probabilities = True
                    else:
                        probabilities = False
                    if 'exact' in dir_name:
                        sigmoid_approx = False
                    else:
                        sigmoid_approx = True
                else:
                    model_name = dir_name
                    with_regularization = False
                    tau, alpha, sigmoid_approx, probabilities = None, None, None, None
                if 'robust' in dir_name:
                    robust_regularization = True
                    beta = float(dir_name.split('_beta_')[1].split('_')[0])
                    gamma = float(dir_name.split('_gamma_')[1].split('_')[0])
                else:
                    robust_regularization = False
                    beta, gamma = None, None


            # for model_name in DATASET_TO_MODEL_NAMES[dataset.split('_')[0].lower()]:
                # taus = DATASET_TO_MODEL_TO_TAUS[dataset.split('_')[0].lower()][model_name]
                # alphas = DATASET_TO_MODEL_TO_ALPHAS[dataset.split('_')[0].lower()][model_name]
                # for (tau_idx, tau), (alpha_idx, alpha) in itertools.product(*[enumerate(taus), enumerate(alphas)]):
                
                regularization_params = {'tau': tau, 'alpha': alpha, 'sigmoid_approx': sigmoid_approx, 
                    'probabilities': probabilities, 'robust_regularization': robust_regularization, 
                    'beta': beta, 'gamma': gamma, 'device': device}
                model_to_load = model.DNN(model_name=model_name, num_classes=ds_obj.num_classes(), 
                    learning_rate=learning_rate, aggregate_coeff=aggregate_coeff,
                    with_regularization=with_regularization, 
                    regularization_params=regularization_params)
                
                complete_model_name = '{}_{}'.format(model_to_load.model_name, model_to_load.criterion._get_name()) \
                    if not isinstance(model_to_load.criterion, nn.CrossEntropyLoss) else model_to_load.model_name

                print ('Attack: {}, Dataset: {}, Model: {}'.format(attack_name, dataset, complete_model_name))

                adv_folder = '../{}/adversarial_images/{}/{}'.format(ds_obj.name, 
                    complete_model_name, attack_name)
                adv_image_ids, all_adv_objs = hp.load_adversarial_objects(folder=adv_folder, epoch=epoch, ds_obj=ds_obj, device=device)
                all_images_adversarial = [x.image for x in all_adv_objs]

                print (adv_folder)
                print (len(glob.glob("{}/*_epoch_{}*".format(adv_folder, epoch))))

                if 'cifar' in ds_obj.name.lower():
                    if ds_obj.name.lower() == 'cifar10':
                        sensitive_attrs, sensitive_attrs_names = [], []
                        for cname in ds_obj.classes:
                            sensitive_attrs_names.append(cname)
                            sensitive_attrs.append(np.array([1 if ds_obj.classes[ds_obj.test_labels[int(img_id)]] == cname \
                                                            else 0 for img_id in adv_image_ids]))
                    else:
                        sensitive_attrs = [np.array(
                                                [1 if ds_obj.classes[ds_obj.test_labels[int(img_id)]] == ds_obj.name.split('_')[-1].lower() \
                                                else 0 for img_id in adv_image_ids])]
                        sensitive_attrs_names = [ds_obj.name.lower().split('_')[-1]]
                else:
                    attr = ds_obj.name.lower().split('_')[-1]
                    sensitive_attrs = [np.array([ds_obj.get_image_protected_class('test', int(img_id), attr=attr) \
                                            for img_id in adv_image_ids])] # sens_attr = 1 means minority
                    sensitive_attrs_names = ['Black' if attr == 'race' else 'Female']

                majority_differences, minority_differences = [], []
                for sensitive_attr in sensitive_attrs:
                    minority_difference, majority_difference = image_differences(adv_image_ids, all_images_adversarial, sensitive_attr, ds_obj)
                    majority_differences.append(majority_difference)
                    minority_differences.append(minority_difference)     

                for minority_difference, majority_difference, sensitive_attr_name in zip(minority_differences, majority_differences, sensitive_attrs_names):
                    mu_minority, mu_majority = np.mean(minority_difference), np.mean(majority_difference)
                    csv_rows.append([dataset, complete_model_name, sensitive_attr_name, mu_minority, mu_majority])
        
        hp.create_dir("pickled_ubs")
        df = pd.DataFrame(csv_rows, columns=['dataset', 'model', 'minority', 'mu_min', 'mu_maj'])
        df.to_csv('pickled_ubs/{}_cdf_mus_regularized.csv'.format(attack_name), index=False)

        print ('Saved to pickled_ubs/{}_cdf_mus_regularized.csv!'.format(attack_name))


def image_differences(adv_image_ids, all_adv_images, sensitive_attr, ds_obj):
    minority_differences, majority_differences = [], []
    for idx, img_id in enumerate(adv_image_ids):
        processed_img = ds_obj.get_image('test', int(img_id))
        raw_img = hp.inverse_transpose_images(processed_img.numpy(), ds_obj.data_transform)
        adv_img = np.moveaxis(all_adv_images[idx], 0, -1) # channels first, non normalized
        if sensitive_attr[idx] == 1:
            minority_differences.append(np.linalg.norm(raw_img - adv_img))
        else:
            majority_differences.append(np.linalg.norm(raw_img - adv_img))
            
    return minority_differences, majority_differences


if __name__=="__main__":
    args_dict = {}
    try:
        gnu_options = ["all_datasets=", "gpu=", "epoch="]
        arguments, values = getopt.getopt(sys.argv[1:], "", gnu_options)

        for cur_arg, cur_val in arguments:
            if 'all_datasets' in cur_arg:
                args_dict[cur_arg.lstrip('--')] = cur_val.split(',')
            else:
                args_dict[cur_arg.lstrip('--')] = cur_val
    except:
        """
        While running test for with_reason, one must run this code separately for all values of num_unshared_layers_idx. 
        Caution: test code must be run only for with_reason and only_classification and must be run after all models have finished training

        Example command to run this file: python experiment.py train only_classification 0:1 True 0 deep_cnn 0 1
        num_unshared_idx is an integer for test but it's a string of the form x:y where x is the start and y is the end index of hp.NUM_UNSHARED_LAYERS

        if --epochs is specified then the value in helper.py is overriden

        dataset = CIFAR10, CIFAR10_regularized_cat, CIFAR10_regularized_automobile
        """
        # raise ValueError("Usage: python experiment.py test/train only_classification/with_reason [<num_unshared_layers_idx_slice>] True/False <gpu_number> [<model_name> <start_idx> <end_idx> ]")
        raise ValueError("""Usage: python experiment.py --all_datasets=<dataset_names comma separated> --gpu=<gpu> --epoch=<which epoch>""")
    main(**args_dict)
