#!/usr/bin/env python
# coding: utf-8

# In[1]:

import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np

import helper as hp

# In[ ]:


class RegularizedLoss:
    
    def __init__(self, alpha=0.005, tau=1.0, sigmoid_approx=True, probabilities=True, 
            robust_regularization=False, beta=None, gamma=None, device='cpu'):
        
        self.alpha = alpha
        self.tau = tau
        self.device = device
        self.sigmoid_approx = sigmoid_approx
        self.probabilities = probabilities
        self.robust_regularization = robust_regularization
        
        if self.probabilities:
            self.name = 'RegularizedLoss_probabilities'
        else:
            self.name = 'RegularizedLoss_distances'
        if self.sigmoid_approx:
            self.name += '_sigmoid_approx'
        else:
            self.name += '_exact'

        self.name += '_alpha_{}_tau_{}'.format(self.alpha, self.tau)

        if self.robust_regularization:
            assert beta is not None and gamma is not None
            self.beta = beta
            self.gamma = gamma
            self.name += '_robust_beta_{}_gamma_{}'.format(self.beta, self.gamma)

        print (self.name)

        self.regularization_terms_batch_train, self.cross_entropy_losses_batch_train, self.total_loss_batch_train = [], [], []
        self.regularization_terms_batch_test, self.cross_entropy_losses_batch_test, self.total_loss_batch_test = [], [], []
        self.regularization_terms_epoch_train, self.cross_entropy_losses_epoch_train, self.total_loss_epoch_train = [], [], []
        self.regularization_terms_epoch_test, self.cross_entropy_losses_epoch_test, self.total_loss_epoch_test = [], [], []
        self.d_approx_majority_test, self.d_approx_minority_test = [], []
        self.d_approx_majority_train, self.d_approx_minority_train = [], []
        self.d_approx_majority_epoch_train, self.d_approx_minority_epoch_train = [], []
        self.d_approx_majority_epoch_test, self.d_approx_minority_epoch_test = [], []

    def _get_name(self):
        return self.name
    
    def __call__(self, outputs, labels, protected_classes, inputs, phase):
        cross_entropy_loss = nn.CrossEntropyLoss()(outputs, labels)
        assert len(inputs) == len(outputs)
        # inputs.requires_grad = True # this needs to be done before having outputs
        # 1 is minority class; 0 is majority
        assert len(protected_classes[protected_classes == -1]) == 0 # nothing should be -1
        # assert len(protected_classes[protected_classes == 1]) > 0
        # assert len(protected_classes[protected_classes == 0]) > 0
        assert (len(protected_classes[protected_classes == 0]) + 
            len(protected_classes[protected_classes == 1])) == len(protected_classes)
        
        _, predicted_classes = torch.max(outputs, 1)
        mask_correct_predictions = predicted_classes == labels
        
        mask_minority = mask_correct_predictions & (protected_classes == 1)
        mask_majority = mask_correct_predictions & (protected_classes == 0)
        minority_outputs, majority_outputs = outputs[mask_minority], outputs[mask_majority]
        
        assert len(minority_outputs) == torch.sum(mask_minority) and \
            len(majority_outputs) == torch.sum(mask_majority)

        if len(minority_outputs) == 0 or len(majority_outputs) == 0:
            ce_loss = cross_entropy_loss.item()
            if phase == 'train':
                self.regularization_terms_batch_train.append(0.)
                self.cross_entropy_losses_batch_train.append(ce_loss)
                self.total_loss_batch_train.append(ce_loss)
            elif phase == 'test':
                self.regularization_terms_batch_test.append(0.)
                self.cross_entropy_losses_batch_test.append(ce_loss)
                self.total_loss_batch_test.append(ce_loss)
            return cross_entropy_loss

        indices_minority = [coord for coord in zip(*enumerate(torch.argmax(minority_outputs, 1)))]
        indices_majority = [coord for coord in zip(*enumerate(torch.argmax(majority_outputs, 1)))]
        
        assert len(indices_majority) == 2 and len(indices_minority) == 2
        
        output_class_logits_minority = minority_outputs[indices_minority[0], indices_minority[1]]
        output_class_logits_majority = majority_outputs[indices_majority[0], indices_majority[1]]
        
        grad_minority = autograd.grad(outputs=output_class_logits_minority, inputs=inputs, 
                             only_inputs=True, retain_graph=True, 
                             grad_outputs=torch.ones_like(output_class_logits_minority, 
                                                          device=self.device))[0][mask_minority]
        grad_majority = autograd.grad(outputs=output_class_logits_majority, inputs=inputs, 
                             only_inputs=True, retain_graph=True, 
                             grad_outputs=torch.ones_like(output_class_logits_majority, 
                                                          device=self.device))[0][mask_majority]
        
        d_approx_minority = torch.abs(output_class_logits_minority).float()/torch.norm(
            grad_minority.view(grad_minority.shape[0], -1), dim=1)
        d_approx_majority = torch.abs(output_class_logits_majority).float()/torch.norm(
            grad_majority.view(grad_majority.shape[0], -1), dim=1)
        
        print (d_approx_minority.shape, d_approx_majority.shape,
            torch.mean(d_approx_minority).item(), torch.mean(d_approx_majority).item())

        if self.probabilities:
            if self.sigmoid_approx:
                # This takes a sigmoid approximation
                regularization_minority = torch.sum(hp.sigmoid(-d_approx_minority + self.tau)).float()/torch.sum(mask_minority)
                regularization_majority = torch.sum(hp.sigmoid(-d_approx_majority + self.tau)).float()/torch.sum(mask_majority)
            else:
                # This does the actual thresholding on tau to calculate exact probabilities
                # (Highly non-smooth and non-differentiable)
                regularization_minority = torch.sum(d_approx_minority < self.tau).float()/torch.sum(mask_minority)
                regularization_majority = torch.sum(d_approx_majority < self.tau).float()/torch.sum(mask_majority)
        else:
            regularization_minority = torch.mean(d_approx_minority[d_approx_minority < self.tau])
            regularization_majority = torch.mean(d_approx_majority[d_approx_majority < self.tau])
        
        # normalize this since CrossEntropyLoss is also normalized
        regularization = torch.abs(regularization_minority - regularization_majority)
        
        if phase == 'train':
            self.regularization_terms_batch_train.append(regularization.item())
            self.cross_entropy_losses_batch_train.append(cross_entropy_loss.item())
            self.total_loss_batch_train.append((cross_entropy_loss + self.alpha * regularization).item())
            self.d_approx_majority_train.extend([x.item() for x in d_approx_majority])
            self.d_approx_minority_train.extend([x.item() for x in d_approx_minority])
        elif phase == 'test':
            self.regularization_terms_batch_test.append(regularization.item())
            self.cross_entropy_losses_batch_test.append(cross_entropy_loss.item())
            self.total_loss_batch_test.append((cross_entropy_loss + self.alpha * regularization).item())
            self.d_approx_majority_test.extend([x.item() for x in d_approx_majority])
            self.d_approx_minority_test.extend([x.item() for x in d_approx_minority])
        
        if self.robust_regularization:
            ## This is the case when we want to reduce unfairness and also increase robustness
            # negative sign since we want to maximize these individual robustness measures of majority and minority

            if self.probabilities and not self.sigmoid_approx:
                assert regularization_majority >= 0 and regularization_minority >= 0

            print ('CE Loss: {}, regularization: {}, regularization_minority: {}, regularization_majority: {}'.format(
                cross_entropy_loss, regularization, regularization_minority, regularization_majority
                ))

            final_loss = cross_entropy_loss + self.alpha * regularization + \
                self.beta * regularization_majority + self.gamma * regularization_minority

        else:
            final_loss = cross_entropy_loss + self.alpha * regularization

        return final_loss
    
    def aggregate_stats(self, phase, verbose=False):
#         self.regularization_terms_batch = [x for x in self.regularization_terms_batch]
#         self.cross_entropy_losses_batch = [x for x in self.cross_entropy_losses_batch]
        
        if phase == 'train':
            self.regularization_terms_epoch_train.append(np.mean(self.regularization_terms_batch_train))
            self.cross_entropy_losses_epoch_train.append(np.mean(self.cross_entropy_losses_batch_train))
            self.total_loss_epoch_train.append(np.mean(self.total_loss_batch_train))
            
            self.d_approx_majority_epoch_train.append(np.mean(self.d_approx_majority_train))
            self.d_approx_minority_epoch_train.append(np.mean(self.d_approx_minority_train))

            (self.regularization_terms_batch_train, self.cross_entropy_losses_batch_train, 
                self.total_loss_batch_train, self.d_approx_majority_train, 
                self.d_approx_minority_train) = [], [], [], [], []
            if verbose:
                print ("Total Loss: {}, Regularization: {}, Cross-Entropy Loss: {}".format(
                    self.total_loss_epoch_train[-1], self.regularization_terms_epoch_train[-1], 
                    self.cross_entropy_losses_epoch_train[-1]))
        elif phase == 'test':
            self.regularization_terms_epoch_test.append(np.mean(self.regularization_terms_batch_test))
            self.cross_entropy_losses_epoch_test.append(np.mean(self.cross_entropy_losses_batch_test))
            self.total_loss_epoch_test.append(np.mean(self.total_loss_batch_test))
            
            self.d_approx_majority_epoch_test.append(np.mean(self.d_approx_majority_test))
            self.d_approx_minority_epoch_test.append(np.mean(self.d_approx_minority_test))

            (self.regularization_terms_batch_test, self.cross_entropy_losses_batch_test, 
                self.total_loss_batch_test, self.d_approx_majority_test, 
                self.d_approx_minority_test) = [], [], [], [], []
            if verbose:
                print ("Total Loss: {}, Regularization: {}, Cross-Entropy Loss: {}".format(
                    self.total_loss_epoch_test[-1], self.regularization_terms_epoch_test[-1], 
                    self.cross_entropy_losses_epoch_test[-1]))

