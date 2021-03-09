import matplotlib as mpl
mpl.use('Agg')

import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score, classification_report

import sys, os
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
learning_rate = 0.01
aggregate_coeff = 5


def main(dataset, gpu, model_names, epochs, taus=[None], alphas=[None], with_regularization=False, 
    sigmoid_approx=False, probabilities=False, robust_regularization=False, betas=[None], gammas=[None]):

    device = torch.device('cuda:{}'.format(gpu))

    attack_names = ['DeepFool', 'CarliniWagner']
    attack_fractions = [0.1, 0.1]
    # attack_kwargs = [{'steps': 100}, {'steps': 100, 'random_start': True}]
    attack_kwargs = [{}] #, {}]
    # Caution: these epsilon values are only for CIFAR10
    attack_call_kwargs = [{}] # , {}]
    # [{attack_name: (param_1, param_2 ....), attack_name: (param_1, param_2 ....),}, {}]
    
    ds_obj, datasets, data_loaders = \
        hp.get_data_loder_objects(dataset, PHASES, **hp.get_loader_kwargs(batch_size))

    for epoch in epochs:
        for model_name in model_names:
            for (tau_idx, tau), (alpha_idx, alpha), (beta_idx, beta), (gamma_idx, gamma) in \
                itertools.product(*[enumerate(taus), enumerate(alphas), enumerate(betas), enumerate(gammas)]):
                
                regularization_params = {'tau': tau, 'alpha': alpha, 'sigmoid_approx': sigmoid_approx, 
                    'probabilities': probabilities, 'robust_regularization': robust_regularization, 
                    'beta': beta, 'gamma': gamma, 'device': device}
                model_to_load = model.DNN(model_name=model_name, num_classes=ds_obj.num_classes(), 
                    learning_rate=learning_rate, aggregate_coeff=aggregate_coeff,
                    with_regularization=with_regularization, 
                    regularization_params=regularization_params)
                
                complete_model_name = '{}_{}'.format(model_to_load.model_name, model_to_load.criterion._get_name()) \
                    if not isinstance(model_to_load.criterion, nn.CrossEntropyLoss) else model_to_load.model_name
                filename = '{}_epoch_{}_lr_{}.pth'.format(complete_model_name, epoch, learning_rate)
                model_to_load.model_ft.load_state_dict(torch.load('../{}/model_weights/{}'.format(ds_obj.name, filename),
                                                         map_location=device))
                model_to_load.model_ft.eval()
                print ('Loaded weights from: ../{}/model_weights/{}'.format(ds_obj.name, filename))
                
                for attack_name, attack_fraction, attack_kwarg, attack_call_kwarg in zip(attack_names, 
                                                                      attack_fractions, 
                                                                      attack_kwargs,
                                                                      attack_call_kwargs):
                    attack_class = Attack if '3.0' in foolbox.__version__ else AttackV2
                    attack = attack_class(model_to_load, ds_obj, device, name=attack_name, 
                                    attack_kwargs=attack_kwarg, attack_call_kwargs=attack_call_kwarg)
                    all_images_adversarial, all_adv_preds, adv_image_ids, total, created = attack.generate_images(data_loaders, portion='test', fraction=attack_fraction, epoch=epoch)
                    
                    print ('{}, {} Total: {}, Created: {}'.format(ds_obj.name, attack_name, total, created))
                    
                    all_adv_objects = hp.make_pickleable(all_images_adversarial, all_adv_preds, adv_image_ids)
                    hp.save_objects(all_adv_objects, complete_model_name, 
                        adv_image_ids, ds_obj, attack, epoch=epoch, 
                        post_fix='_epsilon_{}'.format(attack_call_kwarg['epsilon']) if \
                            'epsilon' in attack_call_kwarg and attack_call_kwarg['epsilon'] is not None else '', 
                            root_dir='..')
                    print ('Saved {}'.format(attack_name))

                    if 'cifar' in ds_obj.name.lower():
                        sensitive_attr = np.array(
                            [1 if ds_obj.classes[ds_obj.test_labels[int(img_id)]] == ds_obj.name.split('_')[-1].lower() \
                            else 0 for img_id in adv_image_ids])
                    else:
                        attr = ds_obj.name.lower().split('_')[-1]
                        sensitive_attr = np.array([ds_obj.get_image_protected_class('test', int(img_id), attr=attr) \
                            for img_id in adv_image_ids])

                    minority_difference, majority_difference = image_differences(adv_image_ids, all_images_adversarial, all_adv_preds, sensitive_attr, ds_obj,
                          attack_name, model_to_load.model_name, model_to_load.criterion._get_name(), root_dir='..')

                    hp.create_dir("plots/{}".format(ds_obj.name))
                    hp.create_dir("plots/{}/{}".format(ds_obj.name, model_to_load.model_name))
                    hp.create_dir("plots/{}/{}/{}".format(ds_obj.name, model_to_load.model_name, attack_name))

                    dir_to_save = "plots/{}/{}/{}".format(ds_obj.name, model_to_load.model_name, attack_name)
                    
                    thresholds = np.linspace(0.0, 2.0, 2000)

                    frac_greater_than_tau_majority = np.array([np.sum(majority_difference > t) / len(majority_difference) for t in thresholds])
                    frac_greater_than_tau_minority = np.array([np.sum(minority_difference > t) / len(minority_difference) for t in thresholds])

                    fig = plt.figure()
                    fig.suptitle(r'fraction $d_\theta > \tau$ for {}'.format(ds_obj.name), fontsize=20)
                    ax = fig.add_subplot(111)
                    ax.plot(thresholds, frac_greater_than_tau_majority, color='blue', label='Other Classes')
                    ax.plot(thresholds, frac_greater_than_tau_minority, color='red', label='{}'.format(ds_obj.name.split('_')[-1]))
                    ax.set_xlabel('Distance to Adv. Sample', fontsize=15)
                    ax.set_ylabel(r'$ \widehat{I^\tau_s} $', fontsize=15)
                    plt.legend()

                    plt.savefig('{}/{}_inv_cdf.png'.format(dir_to_save, model_to_load.criterion._get_name()), bbox_inches='tight')
                    plt.show()
                    plt.close()


def image_differences(adv_image_ids, all_adv_images, all_adv_preds, sensitive_attr, ds_obj,
                      attack_name, model_name, criterion_name, root_dir='.'):
    hp.create_dir("{}/{}/".format(root_dir, ds_obj.name))
    hp.create_dir("{}/{}/adversarial_examples".format(root_dir, ds_obj.name))
    hp.create_dir("{}/{}/adversarial_examples/{}".format(root_dir, ds_obj.name, attack_name))
    hp.create_dir("{}/{}/adversarial_examples/{}/{}".format(root_dir, ds_obj.name, attack_name, model_name))
    
    dir_to_save = "{}/{}/adversarial_examples/{}/{}/".format(root_dir, ds_obj.name, attack_name, model_name)
    minority_differences, majority_differences = [], []
    for idx, img_id in enumerate(adv_image_ids):
        processed_img = ds_obj.get_image('test', int(img_id))
        raw_img = hp.inverse_transpose_images(processed_img.numpy(), ds_obj.data_transform)
        adv_img = np.moveaxis(all_adv_images[idx], 0, -1) # channels first, non normalized
        if sensitive_attr[idx] == 1:
            minority_differences.append(np.linalg.norm(raw_img - adv_img))
        else:
            majority_differences.append(np.linalg.norm(raw_img - adv_img))

        if idx < 1:
            concatenated_images = np.concatenate((raw_img, adv_img), axis=1)

            fig = plt.figure()
            if 'cifar' in ds_obj.name.lower():
                fig.suptitle(
                    'Left: Original Image (correctly predicted: {})\nRight: Adversarial Image (predicted: {})'.\
                    format(ds_obj.classes[ds_obj.test_labels[int(img_id)]], ds_obj.classes[int(all_adv_preds[idx])]), 
                    fontsize=20)
            else:
                fig.suptitle(
                    'Left: Original Image (correctly predicted: {})\nRight: Adversarial Image (predicted: {})'.\
                    format(ds_obj.classes[ds_obj.get_image_label('test', int(img_id))], ds_obj.classes[int(all_adv_preds[idx])]), 
                    fontsize=20)
            ax = fig.add_subplot(111)
            ax.axis('off')
            ax.imshow(concatenated_images, interpolation='bilinear')
            plt.savefig('{}/{}_{}.png'.format(dir_to_save, int(img_id), criterion_name), bbox_inches='tight')
            plt.show()
            
    return minority_differences, majority_differences


if __name__=="__main__":
    args_dict = {}
    try:
        gnu_options = ["dataset=", "model_names=", "gpu=", # Required arguments
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
                args_dict[cur_arg.lstrip('--')] = [float(x) for x in cur_val.split(',')]
                print (cur_arg, args_dict[cur_arg.lstrip('--')])
            elif 'model_names' in cur_arg:
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
        raise ValueError("""Usage: python experiment.py --dataset=<dataset_name> --gpu=<gpu> --epochs=<num_epochs> 
            --model_names=[model_name_1, model_name_2, ...] 
            --taus=[tau_1, tau_2, tau_3, ...] --alphas=[alpha1, alpha2, ...] 
            [--with_reguarization] [--sigmoid_approx] [--probabilities]
            [--robust_regularization] [--betas=[beta_1, beta_2, beta_3, ...]] [--gammas=[gamma_1, gamma_2, gamma_3, ...]]""")
    main(**args_dict)
