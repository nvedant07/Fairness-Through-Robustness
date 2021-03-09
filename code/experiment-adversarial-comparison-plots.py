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

def main(dataset_reg, dataset_original, gpu, model_name_reg, model_name_original, 
    epochs, taus, alphas, sigmoid_approx=False, probabilities=False):

    device = torch.device('cuda:{}'.format(gpu))

    attack_names = ['DeepFool', 'CarliniWagner']
    
    ds_obj_original, _, _ = \
        hp.get_data_loder_objects(dataset_original, PHASES, **hp.get_loader_kwargs(batch_size))
    ds_obj_reg, _, _ = \
        hp.get_data_loder_objects(dataset_reg, PHASES, **hp.get_loader_kwargs(batch_size))

    taus = np.linspace(0.0, 2.0, 2000)

    for epoch in epochs:

        model_original = model.DNN(model_name=model_name_original, num_classes=ds_obj_reg.num_classes(), 
                learning_rate=learning_rate, aggregate_coeff=aggregate_coeff,
                with_regularization=False)
        complete_model_name = '{}_{}'.format(model_original.model_name, model_original.criterion._get_name()) \
            if not isinstance(model_original.criterion, nn.CrossEntropyLoss) else model_original.model_name

        for attack_name in attack_names:

            adv_folder = '../{}/adversarial_images/{}/{}'.format(ds_obj_original.name, 
                complete_model_name, attack_name)
            adv_image_ids, all_adv_objs = hp.load_adversarial_objects(folder=adv_folder, epoch=epoch, ds_obj=ds_obj_original, device=device)
            all_images_adversarial = [x.image for x in all_adv_objs]

            print (adv_folder)
            print (len(glob.glob("{}/*_epoch_{}*".format(adv_folder, epoch))))

            if 'cifar' in ds_obj_original.name.lower():
                sensitive_attrs_name = ds_obj_reg.name.split('_')[-1].lower() # get the sens attr name from reg model
                sensitive_attr = np.array([1 if ds_obj_original.classes[ds_obj_original.test_labels[int(img_id)]] == sensitive_attrs_name \
                    else 0 for img_id in adv_image_ids])
            else:
                attr = ds_obj_original.name.lower().split('_')[-1]
                sensitive_attrs_name = 'Black' if attr == 'race' else 'Female'
                sensitive_attr = np.array([ds_obj_original.get_image_protected_class('test', int(img_id), attr=attr) \
                                        for img_id in adv_image_ids])
                
            minority_difference, majority_difference = image_differences(adv_image_ids, all_images_adversarial, sensitive_attr, ds_obj_original)
            frac_greater_than_tau_majority = np.array([np.sum(majority_difference > t) / len(majority_difference) for t in taus])
            frac_greater_than_tau_minority = np.array([np.sum(minority_difference > t) / len(minority_difference) for t in taus])

            all_lines = [[frac_greater_than_tau_majority, frac_greater_than_tau_minority]]
            titles = ['Original']

            for (tau_idx, tau), (alpha_idx, alpha) in itertools.product(*[enumerate(taus), enumerate(alphas)]):
                regularization_params = {'tau': tau, 'alpha': alpha, 'sigmoid_approx': sigmoid_approx, 
                    'probabilities': probabilities, 'device': device}
                model_reg = model.DNN(model_name=model_name_reg, num_classes=ds_obj_reg.num_classes(), 
                    learning_rate=learning_rate, aggregate_coeff=aggregate_coeff,
                    with_regularization=True, 
                    regularization_params=regularization_params)
                
                complete_model_name = '{}_{}'.format(model_reg.model_name, model_reg.criterion._get_name()) \
                    if not isinstance(model_reg.criterion, nn.CrossEntropyLoss) else model_reg.model_name

                adv_folder = '../{}/adversarial_images/{}/{}'.format(ds_obj_reg.name, 
                    complete_model_name, attack_name)
                adv_image_ids, all_adv_objs = hp.load_adversarial_objects(folder=adv_folder, epoch=epoch, ds_obj=ds_obj_reg, device=device)
                all_images_adversarial = [x.image for x in all_adv_objs]

                print (adv_folder)
                print (len(glob.glob("{}/*_epoch_{}*".format(adv_folder, epoch))))

                if 'cifar' in ds_obj_reg.name.lower():
                    sensitive_attrs_name = ds_obj_reg.name.lower().split('_')[-1]
                    sensitive_attr = np.array([1 if ds_obj_reg.classes[ds_obj_reg.test_labels[int(img_id)]] == sensitive_attrs_name \
                        else 0 for img_id in adv_image_ids])
                    partition_name = 'Partition by class: {}'.format(sensitive_attrs_name)
                else:
                    attr = ds_obj_reg.name.lower().split('_')[-1]
                    sensitive_attrs_name = 'Black' if attr == 'race' else 'Female'
                    sensitive_attr = np.array([ds_obj_reg.get_image_protected_class('test', int(img_id), attr=attr) \
                                            for img_id in adv_image_ids])
                    partition_name = 'Partition by {}: {}'.format(attr, sensitive_attrs_name)
                    
                minority_difference, majority_difference = image_differences(adv_image_ids, all_images_adversarial, sensitive_attr, ds_obj_reg)
                frac_greater_than_tau_majority = np.array([np.sum(majority_difference > t) / len(majority_difference) for t in taus])
                frac_greater_than_tau_minority = np.array([np.sum(minority_difference > t) / len(minority_difference) for t in taus])

                all_lines.append([frac_greater_than_tau_majority, frac_greater_than_tau_minority])
                titles.append(r'$\tau = $' + ' {:.2f}, '.format(tau) + r'$\alpha$' + ' = {:.2f}'.format(alpha))

            
            x_label = 'Distance to Adv. Sample' + r' ($\tau$)'
            y_label = r'$ \widehat{I^\tau_s} $'
            filename = 'inv_cdf_{}_comparison'.format(sensitive_attrs_name)
            dir_to_save = "plots/{}/{}/{}".format(ds_obj_reg.name, model_reg.model_name, attack_name)
            hp.line_plots_grid(all_lines, [taus] * len(all_lines), x_label, y_label, filename, titles, 
                partition_name, subfolder=dir_to_save, y_lims=(0,1), columns=len(all_lines))


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
        gnu_options = \
            ["dataset_reg=", "dataset_original=", "model_name_reg=", "model_name_original=", "gpu=",
            "epochs=", "taus=", "alphas=", "sigmoid_approx", "probabilities"]
        arguments, values = getopt.getopt(sys.argv[1:], "", gnu_options)

        for cur_arg, cur_val in arguments:
            if 'sigmoid_approx' in cur_arg:
                args_dict[cur_arg.lstrip('--')] = True
            elif 'probabilities' in cur_arg:
                args_dict[cur_arg.lstrip('--')] = True
            elif 'epochs' in cur_arg:
                if cur_val == 'all':
                    args_dict[cur_arg.lstrip('--')] = np.arange(0, 100, aggregate_coeff)
                else:
                    args_dict[cur_arg.lstrip('--')] = [cur_val]
            elif 'taus' in cur_arg or 'alphas' in cur_arg:
                args_dict[cur_arg.lstrip('--')] = [float(x) if x.lower() != 'none' else None for x in cur_val.split(',')]
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
        raise ValueError("""Usage: python experiment.py --dataset_reg=<dataset_name> --dataset_original=<dataset_name> 
            --model_name_reg=<model name regularized> --model_name_original=<model name original>
            --gpu=<gpu> --epochs=<num_epochs> --taus=[tau_1, tau_2, tau_3, ...]
            --alphas=[alpha1, alpha2, ...] [--sigmoid_approx] [--probabilities]""")
    main(**args_dict)
