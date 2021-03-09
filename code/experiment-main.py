#!/usr/bin/env python
# coding: utf-8

# ## Entry point; use this to run experiments

# In[1]:

import matplotlib as mpl
mpl.use('Agg')

import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score, classification_report

import sys
import time
import operator
import itertools
import numpy as np
sys.path.insert(0, "../util")


# In[2]:


# get_ipython().run_line_magic('run', 'model.ipynb # defines the model in pytorch; also packs the function to train the model')

from model import *
from pytorch_data_loader import *
from data_loader import *
from helper import *
from adversarial import *

# In[3]:


## Load other helper functions and classes
# get_ipython().run_line_magic('run', 'pytorch_data_loader.ipynb # pytorch data loader class;')
# get_ipython().run_line_magic('run', 'helper.ipynb # helper functions like')
# get_ipython().run_line_magic('run', '../util/data_loader.ipynb')
# get_ipython().run_line_magic('run', 'adversarial.ipynb')


# In[4]:

try:
    dataset_name, gpu = sys.argv[1], sys.argv[2]
    # possible dataset names: utkface_gender, utkface_race, CIFAR10
except:
    raise RuntimeError('Usage: python experiment-main.py <dataset_name> <gpu>')


# In[5]:

batch_size = 28
learning_rate = 0.0005
device = torch.device('cuda:{}'.format(gpu))
num_epochs = 50
aggregate_coeff = 5

# downsample_const = 0.9 if 'CIFAR10' in dataset_name else None # How much to shrink class by
downsample_const = None


# In[6]:


# Declare the DataLoader object -- conatins the full dataset
ds_obj_complete, datasets_complete, data_loaders_complete =     get_data_loder_objects(dataset_name, PHASES, **get_loader_kwargs(batch_size))


# In[7]:


# only for CIFAR10
if downsample_const is not None:
    ds_obj_mapping, datasets_mapping, data_loaders_mapping = {}, {}, {}
    for i in range(ds_obj_complete.num_classes()):
        ds_obj_mapping[i], datasets_mapping[i], data_loaders_mapping[i] =             get_data_loder_objects("CIFAR10_{}".format(ds_obj_complete.classes[i]), 
                                   PHASES, **get_loader_kwargs(batch_size))
        ds_obj_mapping[i].cut_class("train", i, downsample_const)


# In[8]:


### ONLY FOR CIFAR10 ###
### ENSURE CLASSES ARE DOWNSAMPLED CORRECTLY ###
# prints nothing if downsampling is done correctly.
# uses downsample_const defined above

if downsample_const is not None:
    # test size of cut class
    for k in range(ds_obj_complete.num_classes()):

        # Checks number of training points and labels match up
        if (len(ds_obj_mapping[k].train_data) != len(ds_obj_mapping[k].train_labels) ):
            print('mismatch in number of points and labels for ds_obj_mapping[{}]'.format(k))

        cut_class_sz = ds_obj_mapping[k].class_size('train', k) 
        uncut_idxs = list(np.arange(10))
        uncut_idxs.remove(k) # k-th index was cut so remove from this list
        not_cut_class_szs = np.asarray([ds_obj_mapping[k].class_size('train', i) for i in uncut_idxs])

        # Ensure no other class is the same size as the downsampled class.
        # if any other class is same size as the downsampled class
        if (not_cut_class_szs != 5000).any(): 
            print('class should have been full size is not')
        if cut_class_sz != round(5000*(1-downsample_const)): # cut class is correctly downsized
            print('class {} was cut but is not correct size'.format(k))

        # prints nothing if everything worked


# ### Define models

# In[9]:


models_to_copies = {} # There will be multiple copies only for CIFAR10, which has many variants of datasets
for model_name in get_model_names(dataset_name):
    models_to_copies[model_name] = [DNN(model_name=model_name, num_classes=ds_obj_complete.num_classes(), 
                                        learning_rate=learning_rate, aggregate_coeff=aggregate_coeff)]

    if downsample_const is not None:
        for i in range(ds_obj_complete.num_classes()):
            # a separate model for each downsampled instance of CIFAR
            models_to_copies[model_name].append(
                DNN(model_name=model_name, num_classes=ds_obj_complete.num_classes(), 
                    learning_rate=learning_rate, aggregate_coeff=aggregate_coeff))


# In[10]:


models_to_copies


# ### Checks if model weights already exist on disk, if not, it trains the model; else loads the weights

# In[75]:


# for model_name in get_model_names(dataset_name):
#     for idx, model in enumerate(models_to_copies[model_name]):
#         ds_obj, datasets, data_loaders = (ds_obj_complete, datasets_complete, data_loaders_complete)             if idx == 0 else (ds_obj_mapping[idx-1], datasets_mapping[idx-1], data_loaders_mapping[idx-1])
        
#         if not os.path.exists('../{}/model_weights/{}_epoch_best_lr_{}.pth'.format(ds_obj.name, 
#                                                                                    model.model_name,
#                                                                                    learning_rate)):    
#             train_acc_history, train_loss_history, train_accuracy_s0, train_accuracy_s1,                 val_acc_history, val_loss_history, val_accuracy_s0, val_accuracy_s1,                    test_acc_history, test_loss_history, test_accuracy_s0, test_accuracy_s1=                         train_model(model, num_epochs, device, data_loaders)
#             persist_model_weights(model, ds_obj, learning_rate, 'best', root_dir='.')
#         else:
#             with torch.no_grad():
#                 train_acc_history, train_acc_history_s0, train_acc_history_s1, train_loss_history =                     load_model_history(model, ds_obj, num_epochs, portion='train', device=device)
#                 test_acc_history, test_acc_history_s0, test_acc_history_s1, test_loss_history =                     load_model_history(model, ds_obj, num_epochs, portion='test', device=device)

#         line_plots([train_acc_history, test_acc_history], np.arange(0, num_epochs, aggregate_coeff), 
#                    x_label="Epoch", y_label="Accuracy", subfolder=ds_obj.name,
#                    filename='{}_train_test_acc.png'.format(model.model_name),
#                    title="Train vs Test Accuracy ({})".format(model.model_name), 
#                    legend_vals=["Train", "Test"])
#         line_plots([train_loss_history, test_loss_history], np.arange(0, num_epochs, aggregate_coeff), 
#                    x_label="Epoch", y_label="Loss", subfolder=ds_obj.name,
#                    filename='{}_train_test_loss.png'.format(model.model_name),
#                    title="Train vs Test Loss ({})".format(model.model_name), 
#                    legend_vals=["Train", "Test"])


# # ### Prints out statistics about model performance

# # In[16]:


# for model_name in get_model_names(dataset_name):
#     for idx, model in enumerate(models_to_copies[model_name]):
#         ds_obj, datasets, data_loaders = (ds_obj_complete, datasets_complete, data_loaders_complete)             if idx == 0 else (ds_obj_mapping[idx-1], datasets_mapping[idx-1], data_loaders_mapping[idx-1])

#         filename = '{}_epoch_best_lr_{}.pth'.format(model.model_name, learning_rate)
#         model.model_ft.load_state_dict(torch.load('../{}/model_weights/{}'.format(ds_obj.name, filename),
#                                                  map_location=device))
#         model.model_ft.eval()
#         model.model_ft = model.model_ft.to(device)
#         with torch.no_grad():
#             predicted_classes, true_classes, protected_classes, loss =                 get_model_predictions(model, data_loaders=data_loaders, portion="test", device=device)
#             predicted_classes_train, true_classes_train, protected_classes_train, loss_train =                 get_model_predictions(model, data_loaders=data_loaders, portion="train", device=device)
#         print ("Dataset name: {}".format(ds_obj.name))
#         print ("Overall test set accuracy: {}".format(accuracy_score(predicted_classes, true_classes)))
#         print ("Overall train set accuracy: {}".format(accuracy_score(predicted_classes_train, 
#                                                                       true_classes_train)))
#         print ("Classfication report:\n {}".format(classification_report(predicted_classes, true_classes, 
#                                                                        target_names=ds_obj.classes)))
#         print ("Test set breakdown: {}".format(
#             get_class_wise_accuracy(predicted_classes, true_classes, ds_obj.classes)))
#         print ("Train set breakdown: {}".format(
#             get_class_wise_accuracy(predicted_classes_train, true_classes_train, ds_obj.classes)))
#         print ()


# ### Prepare and run the attack(s)


# In[ ]:


attack_names = ['DeepFool', 'CarliniWagner'] #, 'Madry']
attack_fractions = [1.0, 1.0] #, 0.05]

# Note that attack_kwargs are only valid for foolbox v3.0.0b1, for v2.4.0 use everything in attack_call_kwargs
### These are for v3.0.0b1
# attack_kwargs = [{'steps': 100}, {'steps': 100, 'random_start': True}, {}]
# # Caution: these epsilon values are only for CIFAR10
# attack_call_kwargs = [{'epsilons': None}, {'epsilons': 3}]

### These are for v2.4.0
attack_kwargs = [{}, {}] #, {'distance': foolbox.distances.Linfinity}]
attack_call_kwargs = [{'steps': 100}, {'max_iterations': 100}] #, {'epsilon': 0.3, 'stepsize': 0.01, 'iterations': 100}]

model_to_adversarial_results = {model_name:[] for model_name in get_model_names(dataset_name)}
# [{attack_name: (param_1, param_2 ....), attack_name: (param_1, param_2 ....),}, {}]
epoch = 'best'
for model_name in get_model_names(dataset_name):
    for idx, model in enumerate(models_to_copies[model_name]):
        ds_obj, datasets, data_loaders = (ds_obj_complete, datasets_complete, data_loaders_complete)             if idx == 0 else (ds_obj_mapping[idx-1], datasets_mapping[idx-1], data_loaders_mapping[idx-1])
        
        filename = '{}_epoch_{}_lr_{}.pth'.format(model.model_name, epoch, learning_rate)
        model.model_ft.load_state_dict(torch.load('../{}/model_weights/{}'.format(ds_obj.name, filename),
                                                 map_location=device))
        print ('Loaded weights from: ../{}/model_weights/{}'.format(ds_obj.name, filename))
        model.model_ft.eval()
        
        mapping = {}
        for attack_name, attack_fraction, attack_kwarg, attack_call_kwarg in zip(attack_names, 
                                                              attack_fractions, 
                                                              attack_kwargs,
                                                              attack_call_kwargs):
            start = time.time()
            attack_class = Attack if '3.0' in foolbox.__version__ else AttackV2
            attack = attack_class(model, ds_obj, device, name=attack_name, 
                            attack_kwargs=attack_kwarg, attack_call_kwargs=attack_call_kwarg)
            all_images_adversarial, all_adv_preds, adv_image_ids, total, created =                 attack.generate_images(data_loaders, portion='test', fraction=attack_fraction, epoch=epoch)
            
            print ('{}, {} Total: {}, Created: {}'.format(ds_obj.name, attack_name, total, created))
            print ('Time Taken: {}'.format(time.time() - start))
            all_adv_objects = make_pickleable(all_images_adversarial, all_adv_preds, adv_image_ids)
            save_objects(all_adv_objects, model.model_name, adv_image_ids, ds_obj, 
                         attack, epoch=epoch, post_fix='_epsilon_{}'.format(attack_call_kwarg['epsilons'] if \
                             'epsilons' in attack_call_kwarg else None), root_dir='..')
            
            mapping[attack_name] = (all_images_adversarial, all_adv_preds, adv_image_ids)
            with open("../{}/{}_{}_epoch_{}{}_stats.txt".format(ds_obj.name, model_name, 
                                            attack.name, epoch, 
                                            '_epsilon_{}'.format(attack_call_kwarg['epsilons'] if \
                                                 'epsilons' in attack_call_kwarg else None)), 'w') as fp:
                fp.write('{}, {} Total: {}, Created: {}\n'.format(ds_obj.name, attack.name, total, created))
                fp.write('Time Taken: {}\n'.format(time.time() - start))
            
        model_to_adversarial_results[model_name].append(mapping)


# In[26]:


def image_differences(adv_image_ids, all_adv_images, all_adv_preds, sensitive_attr, ds_obj,
                      attack_name, model_name, root_dir='.'):
    create_dir("{}/{}/".format(root_dir, ds_obj.name))
    create_dir("{}/{}/adversarial_examples".format(root_dir, ds_obj.name))
    create_dir("{}/{}/adversarial_examples/{}".format(root_dir, ds_obj.name, attack_name))
    create_dir("{}/{}/adversarial_examples/{}/{}".format(root_dir, ds_obj.name, attack_name, model_name))
    
    dir_to_save = "{}/{}/adversarial_examples/{}/{}/".format(root_dir, ds_obj.name, attack_name, model_name)
    minority_differences, majority_differences = [], []
    for idx, img_id in enumerate(adv_image_ids):
        processed_img = ds_obj.get_image('test', int(img_id))
        raw_img = inverse_transpose_images(processed_img.numpy(), ds_obj.data_transform)
        adv_img = np.moveaxis(all_adv_images[idx], 0, -1) # channels first, non normalized
        if sensitive_attr[idx] == 1:
            minority_differences.append(np.linalg.norm(raw_img - adv_img))
        else:
            majority_differences.append(np.linalg.norm(raw_img - adv_img))

        if idx < 1:
            concatenated_images = np.concatenate((raw_img, adv_img), axis=1)

            fig = plt.figure()
            fig.suptitle(
                'Left: Original Image (correctly predicted: {})\nRight: Adversarial Image (predicted: {})'.\
                format(ds_obj.classes[ds_obj.test_labels[int(img_id)]], ds_obj.classes[int(all_adv_preds[idx])]), 
                fontsize=20)
            ax = fig.add_subplot(111)
            ax.axis('off')
            ax.imshow(concatenated_images, interpolation='bilinear')
            plt.savefig('{}/{}.png'.format(dir_to_save, img_id), bbox_inches='tight')
            plt.show()
            
    return minority_differences, majority_differences


# In[27]:


for model_name in get_model_names(dataset_name):
    for model_idx, model in enumerate(models_to_copies[model_name]):
        if downsample_const is not None and model_idx == 0:
            continue

        ds_obj = ds_obj_complete if model_idx == 0 else ds_obj_mapping[model_idx - 1]
        
        for attack_name, attack_call_kwarg in zip(attack_names, attack_call_kwargs):
            print (attack_name)
            all_images_adversarial, all_adv_preds, adv_image_ids =                 model_to_adversarial_results[model_name][model_idx][attack_name]
            if downsample_const is not None:
                sensitive_attrs_names = [ds_obj.name.split('_')[-1].lower()]
                sensitive_attrs = [np.array(
                    [1 if ds_obj.classes[ds_obj.test_labels[int(img_id)]] == ds_obj.name.split('_')[-1].lower() \
                    else 0 for img_id in adv_image_ids])]
            elif ds_obj.name.lower() == 'cifar10':
                sensitive_attrs, sensitive_attrs_names = [], []
                for i in range(ds_obj.num_classes()):
                    sensitive_attrs_names.append(ds_obj.classes[i])
                    sensitive_attrs.append(np.array(
                        [1 if ds_obj.test_labels[int(img_id)] == i else 0 for img_id in adv_image_ids]))
            else:
                sensitive_attrs_names = [ds_obj.get_sens_attr_name()] # this needs to be implemented in ds_obj
                sensitive_attrs = [np.array([ds_obj.get_image_protected_class(self, 'test', int(img_id))                                            for img_id in adv_image_ids])]
            
            for sensitive_attr_name, sensitive_attr in zip(sensitive_attrs_names, sensitive_attrs):
                minority_difference, majority_difference =                     image_differences(adv_image_ids, all_images_adversarial, all_adv_preds, sensitive_attr, ds_obj,
                          attack_name, model.model_name, root_dir='..')

                create_dir("plots/{}".format(ds_obj.name))
                create_dir("plots/{}/{}".format(ds_obj.name, model.model_name))
                create_dir("plots/{}/{}/{}".format(ds_obj.name, model.model_name, attack_name))

                dir_to_save = "plots/{}/{}/{}".format(ds_obj.name, model.model_name, attack_name)

                taus = np.linspace(0.0, 2.0, 2000) if 'deepfool' in attack_name.lower() else                     np.linspace(0.0, 1.0, 2000) if 'carliniwagner' in attack_name.lower() else                     np.linspace(0.0, 0.3, 2000)

                frac_greater_than_tau_majority =                     np.array([np.sum(majority_difference > t) / len(majority_difference) for t in taus])
                frac_greater_than_tau_minority =                     np.array([np.sum(minority_difference > t) / len(minority_difference) for t in taus])

                fig = plt.figure()
                fig.suptitle(r'fraction $d_\theta > \tau$ for {}'.format(ds_obj.name), fontsize=20)
                ax = fig.add_subplot(111)
                ax.plot(taus, frac_greater_than_tau_majority, color='blue', label='Other Classes')
                ax.plot(taus, frac_greater_than_tau_minority, color='red', label='{}'.format(sensitive_attr_name))
                ax.set_xlabel(r'$\tau$', fontsize=15)
                ax.set_ylabel('Fraction', fontsize=15)
                plt.legend()

                plt.savefig('{}/inv_cdf_{}.png'.format(dir_to_save, sensitive_attr_name), bbox_inches='tight')
                plt.show()
                plt.close()


