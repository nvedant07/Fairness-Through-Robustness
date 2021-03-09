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

def main(dataset, gpu, model_name, epochs, taus, alphas, with_regularization=False, sigmoid_approx=False, probabilities=False):

    device = torch.device('cuda:{}'.format(gpu))

    ds_obj, datasets, data_loaders = \
        hp.get_data_loder_objects(dataset, PHASES, **hp.get_loader_kwargs(batch_size))

    for epoch in epochs:
        for (tau_idx, tau), (alpha_idx, alpha) in itertools.product(*[enumerate(taus), enumerate(alphas)]):
            regularization_params = {'tau': tau, 'alpha': alpha, 'sigmoid_approx': sigmoid_approx, 
                'probabilities': probabilities, 'device': device}
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
            
            complete_model_name = '{}_{}'.format(model_to_load.model_name, model_to_load.criterion._get_name()) \
                if not isinstance(model_to_load.criterion, nn.CrossEntropyLoss) else model_to_load.model_name

            predicted_classes, true_classes = None, None
            for _, inputs, labels, _ in data_loaders['test']:            
                inputs = inputs.to(device)
                model_to_load.model_ft = model_to_load.model_ft.to(device)
                outputs = model_to_load.model_ft(inputs.float())
                _, preds = torch.max(outputs, 1)
                predicted_classes = preds.detach().cpu().numpy() if predicted_classes is None else \
                    np.concatenate((predicted_classes, preds.detach().cpu().numpy()))
                true_classes = labels.numpy() if true_classes is None else \
                    np.concatenate((true_classes, labels.numpy()))

            print ("Accuracy for {}: {}".format(complete_model_name, accuracy_score(true_classes, predicted_classes)))



if __name__=="__main__":
    args_dict = {}
    try:
        gnu_options = ["dataset=", "model_name=", "gpu=", # Required arguments
            "epochs=", "taus=", "alphas=", 
            "with_regularization", "sigmoid_approx", "probabilities"] # Required only when in train mode
        arguments, values = getopt.getopt(sys.argv[1:], "", gnu_options)

        for cur_arg, cur_val in arguments:
            if 'with_regularization' in cur_arg:
                args_dict[cur_arg.lstrip('--')] = True
            elif 'sigmoid_approx' in cur_arg:
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
        raise ValueError("""Usage: python experiment.py --dataset=<dataset_name> --gpu=<gpu> --epochs=<num_epochs> --taus=[tau_1, tau_2, tau_3, ...]
            --alphas=[alpha1, alpha2, ...] [--with_reguarization] [--sigmoid_approx] [--probabilities]""")
    main(**args_dict)
