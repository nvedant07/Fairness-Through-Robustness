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

def main(dataset, gpu, model_name, epochs, taus, alphas, with_regularization=False, 
    sigmoid_approx=False, probabilities=False, robust_regularization=False, betas=[None], gammas=[None]):

    device = torch.device('cuda:{}'.format(gpu))

    attack_names = ['DeepFool', 'CarliniWagner']
    
    ds_obj, datasets, data_loaders = \
        hp.get_data_loder_objects(dataset, PHASES, **hp.get_loader_kwargs(batch_size))

    for epoch in epochs:
        for (tau_idx, tau), (alpha_idx, alpha), (beta_idx, beta), (gamma_idx, gamma) in \
            itertools.product(*[enumerate(taus), enumerate(alphas), enumerate(betas), enumerate(gammas)]):
            
            regularization_params = {'tau': tau, 'alpha': alpha, 'sigmoid_approx': sigmoid_approx, 
                'probabilities': probabilities, 'robust_regularization': robust_regularization, 
                'beta': beta, 'gamma': gamma, 'device': device}
            model_to_load = model.DNN(model_name=model_name, num_classes=ds_obj.num_classes(), 
                learning_rate=learning_rate, aggregate_coeff=aggregate_coeff,
                with_regularization=with_regularization, 
                regularization_params=regularization_params)
            
            # filename = '{}_{}_epoch_{}_lr_{}.pth'.format(model_to_load.model_name, model_to_load.criterion._get_name(), 
            #     epoch, learning_rate)
            # model_to_load.model_ft.load_state_dict(torch.load('../{}/model_weights/{}'.format(ds_obj.name, filename),
            #                                          map_location=device))
            # model_to_load.model_ft.eval()
            # print ('Loaded weights from: ../{}/model_weights/{}'.format(ds_obj.name, filename))
            
            complete_model_name = '{}_{}'.format(model_to_load.model_name, model_to_load.criterion._get_name()) \
                if not isinstance(model_to_load.criterion, nn.CrossEntropyLoss) else model_to_load.model_name

            for attack_name in attack_names:

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
                                            for img_id in adv_image_ids])]
                    sensitive_attrs_names = ['Black' if attr == 'race' else 'Female']

                majority_differences, minority_differences = [], []
                for sensitive_attr in sensitive_attrs:
                    minority_difference, majority_difference = image_differences(adv_image_ids, all_images_adversarial, sensitive_attr, ds_obj)
                    majority_differences.append(majority_difference)
                    minority_differences.append(minority_difference)


                # print (minority_difference, majority_difference)

                hp.create_dir("plots/{}".format(ds_obj.name))
                hp.create_dir("plots/{}/{}".format(ds_obj.name, model_to_load.model_name))
                hp.create_dir("plots/{}/{}/{}".format(ds_obj.name, model_to_load.model_name, attack_name))

                dir_to_save = "plots/{}/{}/{}".format(ds_obj.name, model_to_load.model_name, attack_name)
                
                # taus = np.linspace(0.0, 0.5, 2000)
                taus = np.linspace(0.0, 2.0, 2000)
                # taus = np.linspace(0.0, 2.0, 2000) if 'deepfool' in attack_name.lower() else np.linspace(2.9, 3.1, 2000)

                for minority_difference, majority_difference, sensitive_attr_name in zip(minority_differences, majority_differences, sensitive_attrs_names):
                    frac_greater_than_tau_majority = np.array([np.sum(majority_difference > t) / len(majority_difference) for t in taus])
                    frac_greater_than_tau_minority = np.array([np.sum(minority_difference > t) / len(minority_difference) for t in taus])

                    if paper_friendly_plots:
                        set_paper_friendly_plots_params()

                    fig = plt.figure()
                    if not paper_friendly_plots:
                        fig.suptitle(r'fraction $d_\theta > \tau$ for {}'.format(ds_obj.name), fontsize=20)
                    ax = fig.add_subplot(111)
                    ax.plot(taus, frac_greater_than_tau_majority, color='blue', label='Other Classes')
                    ax.plot(taus, frac_greater_than_tau_minority, color='red', label='{}'.format(sensitive_attr_name))
                    ax.set_xlabel('Distance to Adv. Sample' + r' ($\tau$)')
                    ax.set_ylabel(r'$ \widehat{I^\tau_s} $')
                    plt.legend()

                    extension = 'png' if not paper_friendly_plots else 'pdf'
                    filename = '{}_inv_cdf'.format(model_to_load.criterion._get_name()) \
                        if not isinstance(model_to_load.criterion, nn.CrossEntropyLoss) else \
                            'inv_cdf_{}'.format(sensitive_attr_name)
                    plt.savefig('{}/{}.{}'.format(dir_to_save, filename, extension), bbox_inches='tight')
                    plt.show()
                    plt.close()


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
        gnu_options = ["dataset=", "model_name=", "gpu=", # Required arguments
            "epochs=", "taus=", "alphas=", 
            "with_regularization", "sigmoid_approx", "probabilities",
            "robust_regularization", "betas=", "gammas="] # Required only when in train mode
        arguments, values = getopt.getopt(sys.argv[1:], "", gnu_options)

        for cur_arg, cur_val in arguments:
            if 'with_regularization' in cur_arg:
                args_dict[cur_arg.lstrip('--')] = True
            elif 'sigmoid_approx' in cur_arg:
                args_dict[cur_arg.lstrip('--')] = True
            elif 'probabilities' in cur_arg:
                args_dict[cur_arg.lstrip('--')] = True
            elif 'robust_regularization' in cur_arg:
                args_dict[cur_arg.lstrip('--')] = True
            elif 'epochs' in cur_arg:
                if cur_val == 'all':
                    args_dict[cur_arg.lstrip('--')] = np.arange(0, 100, aggregate_coeff)
                else:
                    args_dict[cur_arg.lstrip('--')] = [cur_val]
            elif 'taus' in cur_arg or 'alphas' in cur_arg or 'betas' in cur_arg or 'gammas' in cur_arg:
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
        raise ValueError("""Usage: python experiment.py --dataset=<dataset_name> --gpu=<gpu> --model_name=<model_name>
            --epochs=<num_epochs> --taus=[tau_1, tau_2, tau_3, ...] --alphas=[alpha1, alpha2, ...] 
            [--with_reguarization] [--sigmoid_approx] [--probabilities]
            [--robust_regularization] [--betas=[beta_1, beta_2, beta_3, ...]] [--gammas=[gamma_1, gamma_2, gamma_3, ...]]""")
    main(**args_dict)
