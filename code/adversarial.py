#!/usr/bin/env python
# coding: utf-8

# ## Defines Adversarial Attacks and defines functions to generate adversarial samples given original inputs and a trained model

# In[1]:

import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim

import sys, os
import time
import operator
import itertools
import numpy as np
import pickle

import foolbox
print (foolbox.__version__)
from foolbox.criteria import Misclassification
from functools import partial
import torch.multiprocessing as multiprocessing

import helper as hp

# In[2]:


dir(foolbox.attacks)


# In[2]:


def get_attack_callable_v2(attack_name, ds, model, criterion, attack_kwargs):
    ## v2 requires only the create_attack_fn such that attack can be created later by calling it, 
    ## no kwargs allowed
    if "fgsm" in attack_name.lower():
        return foolbox.attacks.FGSM(model=model, criterion=criterion, **attack_kwargs)
    if "lbfgs" in attack_name.lower():
        return foolbox.attacks.LBFGSAttack(model=model, criterion=criterion, **attack_kwargs)
    if "deepfool" in attack_name.lower():
        return foolbox.attacks.DeepFoolL2Attack(model=model, criterion=criterion, **attack_kwargs)
    if "madry" in attack_name.lower():
        # L_inf attack
        return foolbox.attacks.RandomStartProjectedGradientDescentAttack(model=model, criterion=criterion, 
                                                                         **attack_kwargs)
    if "carliniwagner" in attack_name.lower():
        return foolbox.attacks.CarliniWagnerL2Attack(model=model, criterion=criterion, **attack_kwargs)


# In[ ]:


def get_attack_callable(attack_name, ds, attack_kwargs, device=None):
    if "fgsm" in attack_name.lower():
        return foolbox.attacks.FGSM(**attack_kwargs)
    if "lbfgs" in attack_name.lower():
        return foolbox.attacks.LBFGSAttack(**attack_kwargs)
    if "deepfool" in attack_name.lower():
        return foolbox.attacks.L2DeepFoolAttack(**attack_kwargs)
    if "madry" in attack_name.lower():
        return foolbox.attacks.L2ProjectedGradientDescentAttack(**attack_kwargs)
    if "carliniwagner" in attack_name.lower():
        return foolbox.attacks.L2CarliniWagnerAttack(**attack_kwargs)


# In[6]:


class Attack:
    """
    Attack class packs in all functions needed to do adversarial attacks
    works for foolbox v3.0.0b1
    """
    def __init__(self, model, ds, device, name='FGSM', 
                 root_dir='.', attack_call_kwargs={}, attack_kwargs={}):
        """
        attack_call_kwargs are the arguments passed as kwargs to the call of the attack. 
            Eg: when doing self.attack(inputs, labels, **attack_call_kwargs)
        attack_kwargs are the arguments passed as kwargs to the definition of the attack object.
            Eg: when doing get_attack_callable(attack_name, fmodel, ds, device, **attack_kwargs)
            Note: this might be used in the future but isn't being used right now
        """
        self.root_dir = root_dir
        self.name = name
        self.ds = ds
        self.device = device
        self.model = model
        self.model.model_ft = self.model.model_ft.double().to(device)
        self.mean = torch.tensor(self.ds.data_transform.transforms[-1].mean, device=self.device).view((3,1,1))
        self.std = torch.tensor(self.ds.data_transform.transforms[-1].std, device=self.device).view((3,1,1))
        self.fmodel = foolbox.models.PyTorchModel(model=self.model.model_ft.double(), 
                                                  bounds=(-0.0, 1.0),
#                                                   num_classes=self.ds.num_classes(), 
                                                  preprocessing={'mean': self.mean.double(), 
                                                                 'std': self.std.double() },
#                                                   preprocessing=None, 
                                                  device=device, 
#                                                   channel_axis=0
                                                 )
        self.attack = get_attack_callable(self.name, self.ds, attack_kwargs, self.device)
        self.attack_call_kwargs = attack_call_kwargs
    
    def generate_images(self, data_loaders, portion, fraction, epoch):
        all_images_adversarial, all_adv_preds, adv_image_ids = None, np.array([]), np.array([])
        total_possible_adv, created_adv = 0, 0
        for idx, (image_ids, inputs, labels, protected_class) in enumerate(data_loaders[portion]):
            print ('Epoch: {}'.format(idx))
            
            image_ids, inputs, labels, protected_class = self.subsample(image_ids, inputs, labels, 
                                                                        protected_class, fraction)
            
            indices_to_consider, all_images_adversarial, all_adv_preds, adv_image_ids =                 self.load_from_disk(image_ids, inputs, labels, protected_class, 
                                    epsilon=self.attack_call_kwargs['epsilons'], epoch=epoch)
            image_ids, inputs, labels, protected_class = (image_ids[indices_to_consider], 
                                                          inputs[indices_to_consider], 
                                                          labels[indices_to_consider], 
                                                          protected_class[indices_to_consider])
            
            inputs, labels, image_ids = inputs.to(self.device), labels.to(self.device), image_ids.to(self.device)
            
            predicted_classes = self.model.model_ft(inputs.double())
            _, predicted_classes = torch.max(predicted_classes, 1)
            
            mask = predicted_classes == labels # only attack correctly classified inputs
            image_ids, inputs, labels, predicted_classes = image_ids[mask], inputs[mask],                 labels[mask], predicted_classes[mask]
            
            # The input taken by the attack is a channels first image, that is not normalized 
            # (It will be mean normalized later on by foolbox. Mean and Std are passed through fmodel)
            inputs_ready_for_attack = hp.inverse_transpose_images(inputs, self.ds.data_transform)
            # because it expects channels first images but not preprocessed
            inputs_ready_for_attack = np.moveaxis(inputs_ready_for_attack, -1, 1)            
            inputs_ready_for_attack = torch.tensor(inputs_ready_for_attack, device=self.device)
            
            ## returned tuple contains 3 elements: 
            ## (perturbed inuts, perturbed inputs clipped to maximum epsilon, 
            ## an array indicating if adversarial attack was a success).
            
            criterion = Misclassification(labels) # untargeted attacks only (for now)
            tup = self.attack(model=self.fmodel, inputs=inputs_ready_for_attack.double(), 
                                           criterion=criterion, **self.attack_call_kwargs)
            
            ## This runs into issues with pickling a pytorch model as defined in foolbox
#             tup = self.parallel_attack(model=self.fmodel, inputs=inputs_ready_for_attack.double(), 
#                                        labels=labels, kwargs=self.attack_call_kwargs)
            
            
            if "deepfool" in self.name:
                ### Epsilon is None for DeepFool, which means that the attacker is 
                ### allowed as much perturbation as needed, so first 2 elements have to be the same
                assert (np.all((tup[0] == tup[1]).cpu().numpy()))
            
            ### these images are NOT normalized and are channels first
            adversarial_images = tup[1]
            ### Sanity Checks
            assert (adversarial_images.shape == inputs.shape)
            for obj in adversarial_images:
                assert obj is not None
            
            # normalize the attacked image for inference, it's already channels first so no need to move axis
            adversarial_images_for_inference = adversarial_images - self.mean
            adversarial_images_for_inference /= self.std
            predictions_on_attacked = self.model(adversarial_images_for_inference.double())
            _, predictions_on_attacked = torch.max(predictions_on_attacked, 1)
            
            """
            places where adversarial attack was a success 
            (this is returned as third element in the tuple by foolbox) but is not really correct in all cases.
            So just to be completely sure, take a bitwise and with what we observe as adversarial.
            """
            adversarial_mask = tup[2] & (predictions_on_attacked != labels)
            total_possible_adv += len(adversarial_mask)
            created_adv += np.count_nonzero(adversarial_mask.cpu().numpy())
            adversarial_images = adversarial_images[adversarial_mask].cpu().numpy()
            predictions_on_attacked = predictions_on_attacked[adversarial_mask].cpu().numpy()
            inputs = inputs[adversarial_mask].cpu().numpy()
            image_ids = image_ids[adversarial_mask].cpu().numpy()
            labels = labels[adversarial_mask].cpu().numpy()
            
            # at this point whatever we have should be adversarial
            assert np.all(labels != predictions_on_attacked)
            
            if len(adversarial_images) == 0:
                continue
            
            ### Visual Sanity Checks
            # these are not normalized, so just need to move the axis
            image_adv = np.moveaxis(adversarial_images[0], 0, -1) 
#             image_adv = hp.inverse_transpose_images(adversarial_images[0], self.ds.data_transform)
            image_original = hp.inverse_transpose_images(inputs[0], self.ds.data_transform)
            stacked_image = np.concatenate((image_adv, image_original), axis=1)
            self.plot_example(stacked_image, labels, predictions_on_attacked)
            
            all_images_adversarial = adversarial_images if all_images_adversarial is None                 else np.concatenate((all_images_adversarial, adversarial_images))
            all_adv_preds = np.concatenate((all_adv_preds, predictions_on_attacked))
            adv_image_ids = np.concatenate((adv_image_ids, image_ids))
        ### adversarial images are channels first, NOT normalized!
        return all_images_adversarial, all_adv_preds, adv_image_ids, total_possible_adv, created_adv
    
    def parallel_attack(self, model, inputs, labels, kwargs, num_slices=10):
        args = [] # the order taken by attack is (model, inputs, criterion)
        slice_size = int(len(inputs)/num_slices)
        for i in range(num_slices):
            s = slice(i * slice_size, (i + 1) * slice_size) if i != num_slices - 1 else                 slice(i * slice_size, len(inputs))
            d = {'model': model, 'inputs': inputs[s], 'criterion': Misclassification(labels[s])}
            for k, v in kwargs.items():
                d[k] = v
            args.append(d)
        with multiprocessing.Pool(processes=num_slices) as pool:
            all_tups = starmap_with_kwargs(pool, self.attack, args)
#             all_tups = pool.starmap(partial(self.attack(**kwargs)), args)
        return concatenate_tup_entries(all_tups)
    
    def subsample(self, image_ids, inputs, labels, protected_class, fraction):
        subsampled_indices = list(range(int(len(image_ids) * fraction)))
        for protected_class_idx in np.where(protected_class == 1)[0]:
            if protected_class_idx not in subsampled_indices:
                subsampled_indices.append(protected_class_idx)

        image_ids = image_ids[subsampled_indices]
        inputs = inputs[subsampled_indices,:,:,:]
        labels = labels[subsampled_indices]
        protected_class = protected_class[subsampled_indices]
        
        return image_ids, inputs, labels, protected_class
    
    def plot_example(self, stacked_image, labels, predictions_on_attacked):
        if not os.path.exists('adversarial_examples/{}/{}.png'.format(self.model.model_name, 
                                                                      '{}_adv_example'.format(self.name))):
            plot_image(stacked_image, subfolder=self.model.model_name, 
                   filename='{}_adv_example'.format(self.name), 
                   plot_title='True Class: {}, Pred Class: {}'.format(
                       self.ds.classes[labels[0]], self.ds.classes[predictions_on_attacked[0]]))
    
    def load_from_disk(self, image_ids, inputs, labels, protected_class, epsilon, epoch):
        image_ids_found, indices = [], []
        all_images_adversarial, all_adv_preds, adv_image_ids = None, np.array([]), np.array([])
        for idx, image_id in enumerate(image_ids):
            exists, adv_object = self.get_adversarial_image(image_id, epoch, 
                                                            post_fix='epsilon_{}'.format(epsilon))
            if not exists:
                indices.append(idx)
                continue
            all_images_adversarial = adv_object.image if all_images_adversarial is None else                 np.concatenate((all_images_adversarial, adv_object.image))
            all_adv_preds = np.concatenate((all_adv_preds, adv_object.prediction))
            adv_image_ids = np.concatenate((adv_image_ids, image_id))
        
        if all_images_adversarial is not None:        
            adversarial_images_for_inference =                 torch.tensor(all_images_adversarial, device=self.device) - self.mean
            adversarial_images_for_inference /= self.std
            predictions_on_attacked = self.model(adversarial_images_for_inference.double())
            _, predictions_on_attacked = torch.max(predictions_on_attacked, 1)

            assert np.all(predictions_on_attacked.cpu().numpy() == all_adv_preds)
        
            print ("Succesfully loaded {} adversarial images from disk!".format(len(all_images_adversarial)))
        
        return indices, all_images_adversarial, all_adv_preds, adv_image_ids
    
    def get_adversarial_image(self, image_id, epoch, post_fix):
        """
        checks on disk, if found, return the adversarial object
        """
        if not os.path.exists("{}/../{}/adversarial_images/{}/{}".format(self.root_dir, self.ds.name, 
                                                                         self.model.model_name, self.name)):
            return False, None
        else:
            if not os.path.exists("{}/../{}/adversarial_images/{}/{}/{}_epoch_{}{}.pkl".format(self.root_dir, 
                                                    self.ds.name, self.model.model_name, self.name, 
                                                    int(image_id), epoch, post_fix)):
                return False, None
            with open("{}/../{}/adversarial_images/{}/{}/{}_epoch_{}{}.pkl".format(self.root_dir, 
                                                    self.ds.name, self.model.model_name, self.name, 
                                                    int(image_id), epoch, post_fix), 'rb') as handle:
                adv_obj = pickle.load(handle)
            return True, adv_obj


# In[ ]:


class AttackV2:
    """
    Attack class packs in all functions needed to do adversarial attacks
    Designed for foolbox v2.4.0
    """
    def __init__(self, model, ds, device, name='FGSM', 
                 root_dir='.', attack_call_kwargs={}, attack_kwargs={}):
        """
        attack_call_kwargs are the arguments passed as kwargs to the call of the attack. 
            Eg: when doing self.attack(inputs, labels, **attack_call_kwargs)
        attack_kwargs are the arguments passed as kwargs to the definition of the attack object.
            Eg: when doing get_attack_callable(attack_name, fmodel, ds, device, **attack_kwargs)
            Note: this might be used in the future but isn't being used right now
        """
        self.root_dir = root_dir
        self.name = name
        self.ds = ds
        self.device = device
        self.model = model
        self.model.model_ft = self.model.model_ft.double()
        self.mean = np.array(self.ds.data_transform.transforms[-1].mean).reshape((3,1,1))
        self.std = np.array(self.ds.data_transform.transforms[-1].std).reshape((3,1,1))
        self.fmodel = foolbox.models.PyTorchModel(model=self.model.model_ft.double(), 
                                                  bounds=(-0.000001, 1.000001),
                                                  num_classes=self.ds.num_classes(), 
                                                  preprocessing={'mean': self.mean, 
                                                                 'std': self.std },
#                                                   preprocessing=None, 
                                                  device=device, 
                                                  channel_axis=0
                                                 )
        self.criterion = Misclassification()
        self.attack = get_attack_callable_v2(self.name, self.ds, self.fmodel, self.criterion, attack_kwargs)
        self.attack_call_kwargs = attack_call_kwargs

    
    def generate_images(self, data_loaders, portion, fraction, epoch):
        all_images_adversarial, all_adv_preds, adv_image_ids = None, np.array([]), np.array([])
        total_possible_adv, created_adv = 0, 0
        for idx, (image_ids, inputs, labels, protected_class) in enumerate(data_loaders[portion]):
            print ('Epoch: {}'.format(idx))
            print (all_images_adversarial.shape if all_images_adversarial is not None else 0)
            print ()
            
            image_ids, inputs, labels, protected_class = self.subsample(image_ids, inputs, labels, 
                                                                        protected_class, fraction)
            
            indices_to_consider, all_images_adversarial_loaded, all_adv_preds_loaded, adv_image_ids_loaded =                 self.load_from_disk(image_ids, inputs, labels, protected_class, 
                                    epsilon=self.attack_call_kwargs['epsilon'] \
                                    if 'epsilon' in self.attack_call_kwargs else None, epoch=epoch)
            
            if len(adv_image_ids_loaded) > 0:
                all_images_adversarial = all_images_adversarial_loaded if all_images_adversarial is None else                     np.concatenate((all_images_adversarial, all_images_adversarial_loaded))
                all_adv_preds = np.concatenate((all_adv_preds, all_adv_preds_loaded))
                adv_image_ids = np.concatenate((adv_image_ids, adv_image_ids_loaded))
            
            image_ids, inputs, labels, protected_class = (image_ids[indices_to_consider], 
                                                          inputs[indices_to_consider], 
                                                          labels[indices_to_consider], 
                                                          protected_class[indices_to_consider])
            
            if len(image_ids) == 0:
                inputs, labels, image_ids = None, None, None
                torch.cuda.empty_cache()
                continue
            
            inputs, labels, image_ids = inputs.to(self.device), labels.to(self.device), image_ids.to(self.device)
            
            predicted_classes = self.model.model_ft.double()(inputs.double())
            _, predicted_classes = torch.max(predicted_classes, 1)
            
            mask = predicted_classes == labels # only attack correctly classified inputs
            image_ids, inputs, labels, predicted_classes = image_ids[mask], inputs[mask],                 labels[mask], predicted_classes[mask]
            
            if len(image_ids) == 0:
                inputs, labels, image_ids = None, None, None
                torch.cuda.empty_cache()
                continue

            # The input taken by the attack is a channels first image, that is not normalized 
            # (It will be mean normalized later on by foolbox. Mean and Std are passed through fmodel)
            inputs_ready_for_attack = hp.inverse_transpose_images(inputs, self.ds.data_transform)
            # because it expects channels first images but not preprocessed
            inputs_ready_for_attack = np.moveaxis(inputs_ready_for_attack, -1, 1)
#             inputs_ready_for_attack = torch.tensor(inputs_ready_for_attack, device=self.device)
            
            ## returned tuple contains 3 elements: 
            ## (perturbed inuts, perturbed inputs clipped to maximum epsilon, 
            ## an array indicating if adversarial attack was a success).
            
            adv_objects = self.attack(inputs=inputs_ready_for_attack, labels=labels.cpu().numpy(), 
                              **self.attack_call_kwargs, unpack=False)
            total_possible_adv += len(adv_objects)

            ### these images are NOT normalized and are channels first
            # for these indices, no adversarial image was found
            indices_to_include, adversarial_images = np.array([]), None
            for idx, obj in enumerate(adv_objects):
                if obj is not None and obj.perturbed is not None:
                    indices_to_include = np.append(indices_to_include, idx)
                    adversarial_images = obj.perturbed.reshape(1, *obj.perturbed.shape)                         if adversarial_images is None else np.concatenate((adversarial_images, 
                                                                obj.perturbed.reshape(1, *obj.perturbed.shape)))
            
            if adversarial_images is None:
                inputs, labels, image_ids = None, None, None
                torch.cuda.empty_cache()
                continue
            
            inputs, labels, image_ids = (inputs[indices_to_include], labels[indices_to_include], 
                                         image_ids[indices_to_include])
            
            # normalize the attacked image for inference, it's already channels first so no need to move axis
            adversarial_images_for_inference = adversarial_images - self.mean
            adversarial_images_for_inference /= self.std
            adversarial_images_for_inference = torch.tensor(adversarial_images_for_inference, device=self.device)
            with torch.no_grad():
                predictions_on_attacked = self.model(adversarial_images_for_inference.double())
                _, predictions_on_attacked = torch.max(predictions_on_attacked, 1)
            
            adversarial_mask = predictions_on_attacked != labels
            print (np.count_nonzero(adversarial_mask.cpu().numpy()))
            created_adv += np.count_nonzero(adversarial_mask.cpu().numpy())
            adversarial_images = adversarial_images[adversarial_mask.cpu().numpy()]
            predictions_on_attacked = predictions_on_attacked[adversarial_mask].cpu().numpy()
            inputs = inputs[adversarial_mask].cpu().numpy()
            image_ids = image_ids[adversarial_mask].cpu().numpy()
            labels = labels[adversarial_mask].cpu().numpy()
            
            # at this point whatever we have should be adversarial
            assert np.all(labels != predictions_on_attacked)
            
            if len(adversarial_images) == 0:
                inputs, labels, image_ids, adversarial_images_for_inference = None, None, None, None
                torch.cuda.empty_cache()
                continue
            
            ### Visual Sanity Checks
            # these are not normalized, so just need to move the axis
            image_adv = np.moveaxis(adversarial_images[0], 0, -1) 
#             image_adv = hp.inverse_transpose_images(adversarial_images[0], self.ds.data_transform)
            image_original = hp.inverse_transpose_images(inputs[0], self.ds.data_transform)
            stacked_image = np.concatenate((image_adv, image_original), axis=1)
            self.plot_example(stacked_image, labels, predictions_on_attacked)
            
            all_images_adversarial = adversarial_images if all_images_adversarial is None                 else np.concatenate((all_images_adversarial, adversarial_images))
            all_adv_preds = np.concatenate((all_adv_preds, predictions_on_attacked))
            adv_image_ids = np.concatenate((adv_image_ids, image_ids))
            
            inputs, labels, image_ids, adversarial_images_for_inference = None, None, None, None
            
            try:
                torch.cuda.empty_cache()
            except:
                pass
        ### adversarial images are channels first, NOT normalized!
        return all_images_adversarial, all_adv_preds, adv_image_ids, total_possible_adv, created_adv
    
    def parallel_attack(self, model, inputs, labels, kwargs, num_slices=10):
        args = [] # the order taken by attack is (model, inputs, criterion)
        slice_size = int(len(inputs)/num_slices)
        for i in range(num_slices):
            s = slice(i * slice_size, (i + 1) * slice_size) if i != num_slices - 1 else                 slice(i * slice_size, len(inputs))
            d = {'model': model, 'inputs': inputs[s], 'criterion': Misclassification(labels[s])}
            for k, v in kwargs.items():
                d[k] = v
            args.append(d)
        with multiprocessing.Pool(processes=num_slices) as pool:
            all_tups = starmap_with_kwargs(pool, self.attack, args)
#             all_tups = pool.starmap(partial(self.attack(**kwargs)), args)
        return concatenate_tup_entries(all_tups)
    
    def subsample(self, image_ids, inputs, labels, protected_class, fraction):
        subsampled_indices = list(range(int(len(image_ids) * fraction)))
        for protected_class_idx in np.where(protected_class == 1)[0]:
            if protected_class_idx not in subsampled_indices:
                subsampled_indices.append(protected_class_idx)

        image_ids = image_ids[subsampled_indices]
        inputs = inputs[subsampled_indices,:,:,:]
        labels = labels[subsampled_indices]
        protected_class = protected_class[subsampled_indices]
        
        return image_ids, inputs, labels, protected_class
    
    def plot_example(self, stacked_image, labels, predictions_on_attacked):
#         if not os.path.exists('adversarial_examples/{}/{}.png'.format(self.model.model_name, 
#                                                                       '{}_adv_example'.format(self.name))):
        hp.plot_image(stacked_image, subfolder=self.model.model_name, 
               filename='{}_adv_example'.format(self.name), 
               plot_title='True Class: {}, Pred Class: {}'.format(
                   self.ds.classes[labels[0]], self.ds.classes[predictions_on_attacked[0]]))
    
    def load_from_disk(self, image_ids, inputs, labels, protected_class, epsilon, epoch):
        image_ids_found, indices = [], []
        all_images_adversarial, all_adv_preds, adv_image_ids = None, np.array([]), np.array([])
        for idx, image_id in enumerate(image_ids):
            exists, adv_object = self.get_adversarial_image(image_id, epoch, 
                                                            post_fix='_epsilon_{}'.format(epsilon) if epsilon is not None else '')
            if not exists:
                indices.append(idx)
                continue
            loaded_image = adv_object.image.reshape(1, *adv_object.image.shape)
            all_images_adversarial = loaded_image                 if all_images_adversarial is None else np.concatenate((all_images_adversarial, loaded_image))
            all_adv_preds = np.concatenate((all_adv_preds, [adv_object.prediction]))
            adv_image_ids = np.concatenate((adv_image_ids, [image_id]))
            
            adv_object = None
            torch.cuda.empty_cache()
        
        if all_images_adversarial is not None:        
            adversarial_images_for_inference = all_images_adversarial - self.mean
            adversarial_images_for_inference /= self.std
            adversarial_images_for_inference = torch.tensor(adversarial_images_for_inference, device=self.device)
            with torch.no_grad():
                predictions_on_attacked = self.model.model_ft(adversarial_images_for_inference.double())
                _, predictions_on_attacked = torch.max(predictions_on_attacked, 1)
            
            assert np.all(predictions_on_attacked.cpu().numpy() == all_adv_preds)
        
            print ("Succesfully loaded {} adversarial images from disk!".format(len(all_images_adversarial)))
            
            adversarial_images_for_inference = None
            torch.cuda.empty_cache()
        
        return indices, all_images_adversarial, all_adv_preds, adv_image_ids
    
    def get_adversarial_image(self, image_id, epoch, post_fix):
        """
        checks on disk, if found, return the adversarial object
        """
        if not os.path.exists("{}/../{}/adversarial_images/{}_{}/{}".format(self.root_dir, self.ds.name, 
            self.model.model_name, self.model.criterion._get_name(), self.name)):
            return False, None
        else:
            if not os.path.exists("{}/../{}/adversarial_images/{}_{}/{}/{}_epoch_{}{}.pkl".format(self.root_dir, 
                                self.ds.name, self.model.model_name, self.model.criterion._get_name(), 
                                self.name, int(image_id), epoch, post_fix)):
                return False, None
            with open("{}/../{}/adversarial_images/{}_{}/{}/{}_epoch_{}{}.pkl".format(self.root_dir, 
                                self.ds.name, self.model.model_name, self.model.criterion._get_name(), self.name, 
                                int(image_id), epoch, post_fix), 'rb') as handle:
                adv_obj = pickle.load(handle)
            return True, adv_obj


# In[ ]:


