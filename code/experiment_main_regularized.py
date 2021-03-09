#!/usr/bin/env python
# coding: utf-8

# ## This is for regularized loss, integrate into the main file soon

# In[1]:


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
import foolbox
import getopt
sys.path.insert(0, "../util")


import model


## Load other helper functions and classes
from pytorch_data_loader import PytorchLoader
import helper as hp
from data_loader import UTKFace, Adience, CIFAR10
from adversarial import Attack, AttackV2

# In[4]:

PHASES = ['train', 'test']
batch_size = 100
learning_rate = 0.01
aggregate_coeff = 5

def main(dataset, gpu, epochs, model_names, with_regularization=False, taus=None, alphas=None, 
    sigmoid_approx=False, probabilities=False, robust_regularization=False, betas=None, gammas=None):
    
    if with_regularization:
        assert taus is not None and alphas is not None
        if robust_regularization:
            assert betas is not None and gammas is not None
    else:
        taus, alphas, betas, gammas = [None], [None], [None], [None]

    device = torch.device('cuda:{}'.format(gpu))    

    ds_obj, datasets, data_loaders = \
        hp.get_data_loder_objects(dataset, PHASES, **hp.get_loader_kwargs(batch_size))

    for (tau_idx, tau), (alpha_idx, alpha), (beta_idx, beta), (gamma_idx, gamma) in \
        itertools.product(*[enumerate(taus), enumerate(alphas), enumerate(betas), enumerate(gammas)]):

        regularization_params = {'tau': tau, 'alpha': alpha, 'sigmoid_approx': sigmoid_approx, 
            'probabilities': probabilities, 'robust_regularization': robust_regularization, 
            'beta': beta, 'gamma': gamma, 'device': device}
        criterion_kwargs = {} if not with_regularization else {'inputs': None, 'protected_classes': None}
        
        # assert model_name in hp.get_model_names(dataset, with_regularization)
        for model_name in model_names:
            model_to_train = model.DNN(model_name=model_name, num_classes=ds_obj.num_classes(), 
                                learning_rate=learning_rate, aggregate_coeff=aggregate_coeff,
                                with_regularization=with_regularization, 
                                regularization_params=regularization_params)
            if not os.path.exists('../{}/training_values/{}_{}_lr_{}_train_acc_history.pkl'.format(ds_obj.name, 
                model_to_train.model_name, model_to_train.criterion._get_name(), learning_rate)):
                
                (train_acc_history, train_overall_loss_history, train_cross_entropy, 
                    train_regularization, train_minority_dist, train_majority_dist,
                        test_acc_history, test_overall_loss_history, test_cross_entropy, 
                            test_regularization, test_minority_dist, test_majority_dist) = \
                                model.train_model(model_to_train, epochs, device, data_loaders, 
                                    criterion_kwargs)
                print ((train_acc_history, train_overall_loss_history, train_cross_entropy, 
                    train_regularization, train_minority_dist, train_majority_dist,
                        test_acc_history, test_overall_loss_history, test_cross_entropy, 
                            test_regularization, test_minority_dist, test_majority_dist))
                hp.persist_model_weights(model_to_train, ds_obj, learning_rate, 'best', root_dir='.')
                hp.persist_epoch_values(
                    model_to_train, ds_obj, learning_rate,
                    (train_acc_history, train_overall_loss_history, train_cross_entropy, 
                        train_regularization, train_minority_dist, train_majority_dist,
                            test_acc_history, test_overall_loss_history, test_cross_entropy, 
                                test_regularization, test_minority_dist, test_majority_dist),
                    ('train_acc_history', 'train_overall_loss_history', 'train_cross_entropy', 
                        'train_regularization', 'train_minority_dist', 'train_majority_dist',
                            'test_acc_history', 'test_overall_loss_history', 'test_cross_entropy', 
                                'test_regularization', 'test_minority_dist', 'test_majority_dist'))
            else:
                (train_acc_history, train_overall_loss_history, train_cross_entropy, 
                    train_regularization, train_minority_dist, train_majority_dist,
                        test_acc_history, test_overall_loss_history, test_cross_entropy, 
                            test_regularization, test_minority_dist, test_majority_dist) = \
                    hp.load_epoch_values(model_to_train, ds_obj, learning_rate, 
                        ('train_acc_history', 'train_overall_loss_history', 'train_cross_entropy', 
                            'train_regularization', 'train_minority_dist', 'train_majority_dist',
                                'test_acc_history', 'test_overall_loss_history', 'test_cross_entropy', 
                                    'test_regularization', 'test_minority_dist', 'test_majority_dist'))
                # with torch.no_grad():
                    # train_acc_history, train_acc_history_s0, train_acc_history_s1, train_loss_history = model.load_model_history(model, ds_obj, num_epochs, portion='train', device=device, 
                    #                        override_criterion=nn.CrossEntropyLoss())
                    # test_acc_history, test_acc_history_s0, test_acc_history_s1, test_loss_history = model.load_model_history(model, ds_obj, num_epochs, portion='test', device=device, 
                    #                        override_criterion=nn.CrossEntropyLoss())



            hp.line_plots([train_acc_history, test_acc_history], np.arange(0, epochs, aggregate_coeff), 
                x_label="Epoch", y_label="Accuracy", subfolder=ds_obj.name,
                filename='{}_{}_train_test_acc.png'.format(model_to_train.model_name, model_to_train.criterion._get_name()),
                title="Accuracy ({})".format(model_to_train.model_name), 
                legend_vals=["Train", "Test"])
            hp.line_plots([train_overall_loss_history, test_overall_loss_history], np.arange(0, epochs, aggregate_coeff), 
                x_label="Epoch", y_label="Total Loss", subfolder=ds_obj.name,
                filename='{}_{}_train_test_overall_loss.png'.format(model_to_train.model_name, model_to_train.criterion._get_name()),
                title="Overall Loss ({})".format(model_to_train.model_name), 
                legend_vals=["Train", "Test"])
            hp.line_plots([train_cross_entropy, test_cross_entropy], np.arange(0, epochs, aggregate_coeff), 
                x_label="Epoch", y_label="Cross Entropy Loss", subfolder=ds_obj.name,
                filename='{}_{}_train_test_ce_loss.png'.format(model_to_train.model_name, model_to_train.criterion._get_name()),
                title="CE Loss ({})".format(model_to_train.model_name), 
                legend_vals=["Train", "Test"])
            hp.line_plots([train_regularization, test_regularization], np.arange(0, epochs, aggregate_coeff), 
                x_label="Epoch", y_label="Reg. Term", subfolder=ds_obj.name,
                filename='{}_{}_train_test_reg_term.png'.format(model_to_train.model_name, model_to_train.criterion._get_name()),
                title="Reg Term ({})".format(model_to_train.model_name), 
                legend_vals=["Train", "Test"])
            hp.line_plots([train_minority_dist, test_minority_dist], np.arange(0, epochs, aggregate_coeff), 
                x_label="Epoch", y_label="Minority Dist Approx", subfolder=ds_obj.name,
                filename='{}_{}_train_test_minority_dist.png'.format(model_to_train.model_name, model_to_train.criterion._get_name()),
                title="Reg Term ({})".format(model_to_train.model_name), 
                legend_vals=["Train", "Test"])
            hp.line_plots([train_majority_dist, test_majority_dist], np.arange(0, epochs, aggregate_coeff), 
                x_label="Epoch", y_label="Majority Dist Approx", subfolder=ds_obj.name,
                filename='{}_{}_train_test_majority_dist.png'.format(model_to_train.model_name, model_to_train.criterion._get_name()),
                title="Reg Term ({})".format(model_to_train.model_name), 
                legend_vals=["Train", "Test"])



if __name__=="__main__":
    args_dict = {}
    try:
        gnu_options = ["dataset=", "gpu=", # Required arguments
            "epochs=", "taus=", "alphas=", "model_names=",
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
                args_dict[cur_arg.lstrip('--')] = int(cur_val)
            elif 'taus' in cur_arg or 'alphas' in cur_arg or 'betas' in cur_arg or 'gammas' in cur_arg:
                args_dict[cur_arg.lstrip('--')] = [float(x) for x in cur_val.split(',')]
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
        raise ValueError("""Usage: python experiment.py --dataset=<dataset_name> --gpu=<gpu> --epochs=<num_epochs> --taus=[tau_1, tau_2, tau_3, ...] --model_names=[model_name_1, model_name_2, ...]
            --alphas=[alpha1, alpha2, ...] [--with_reguarization] [--sigmoid_approx] [--probabilities] [--robust_regularization] [--betas=[beta_1, beta_2, beta_3, ...]] [--gammas=[gamma_1, gamma_2, gamma_3, ...]]""")
    main(**args_dict)

