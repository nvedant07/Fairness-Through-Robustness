#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim

import copy, time
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Load the Pytorch Dataset class which will be used to iterate over data

from pytorch_data_loader import PytorchLoader
from custom_models import DeepCNN, UTKClassifier, AdienceClassifier
from regularized_loss import RegularizedLoss
import helper as hp


# In[1]:


def set_parameter_requires_grad(model, num_unfrozen=None):
    """
    Function that taken in an instance of DNN (model) and the number of layers to unfreeze.
    If num_unfreeze is None, then all layers are unfrozen.
    """
    model_parameters = list(model.parameters())
    for param in model_parameters:
        param.requires_grad = True if num_unfrozen is None else False
    if num_unfrozen is not None:
        for i in range(1, num_unfrozen + 1):
            model_parameters[-2-i].requires_grad = True


# In[2]:

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if "resnet" in model_name:
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, None)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif "alexnet" in model_name:
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, None)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif "vgg" in model_name:
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, None)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif "squeezenet" in model_name:
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, None)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif "densenet" in model_name:
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, None)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif "inception" in model_name:
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, None)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size




class DNN:
    """
    Defines the model architecture and sets other hyperparams like learning_rate etc
    
    Currently this supports only VGG16 and VGG19.
    """
    def __init__(self, model_name, num_classes, learning_rate, aggregate_coeff, use_pretrained=True, 
                 with_regularization=False, regularization_params=None):
        """
        with_regularization: 
            if True then it trains the model with regularization that ensures both subgroups are equally secure
        regularization_params: 
            only matters when `with_regularization` is set to True
        """
        self.num_classes = num_classes
        self.model_name = model_name
        self.use_pretrained = use_pretrained
        self.learning_rate = learning_rate
        self.aggregate_coeff = aggregate_coeff
        self.criterion = nn.CrossEntropyLoss() if not with_regularization else RegularizedLoss(**regularization_params)
        self.model_ft = self.define_model()
    
    def __call__(self, inputs):
        # if isinstance(self.model_ft.fc3.weight.type(), torch.cuda.FloatTensor):
        #     self.model_ft = self.model_ft.double()
        return self.model_ft.double()(inputs.double())
    
    def define_model(self):
        if self.model_name.lower().startswith("vgg"):
            """ 
            VGG16/VGG19 with batch normalization
            """
            model_ft = models.vgg16_bn(pretrained=self.use_pretrained) if self.model_name.lower() == "vgg16"                else models.vgg19_bn(pretrained=self.use_pretrained)
            set_parameter_requires_grad(model_ft)
            num_ftrs = model_ft.classifier[6].in_features
            self.input_size = 224
            model_ft.classifier[6] = nn.Linear(num_ftrs, self.num_classes)
        elif 'adience_classifier' in self.model_name.lower():
            """
            Adience classifier taken from:
            https://talhassner.github.io/home/projects/cnn_agegender/CVPR2015_CNN_AgeGenderEstimation.pdf
            """
            model_ft = AdienceClassifier(self.num_classes)
            set_parameter_requires_grad(model_ft)
        elif 'utk_classifier' in self.model_name.lower():
            """
            UTKFace classifier taken (mostly) from Kaggle: 
            https://www.kaggle.com/loaiabdalslam/gender-group-classification-with-cnn
            """
            model_ft = UTKClassifier(self.num_classes)
            set_parameter_requires_grad(model_ft)
        elif "deep_cnn" in self.model_name.lower():
            """
            Custom model defined for CIFAR-10 dataset
            """
            model_ft = DeepCNN(self.num_classes)
            set_parameter_requires_grad(model_ft)
        elif "mlp1_cifar" in self.model_name.lower():
            model_ft = MLP1(3072,self.num_classes)
            set_parameter_requires_grad(model_ft)
        elif "mlp1_adience" in self.model_name.lower():
            model_ft = MLP1(150528,self.num_classes)
            set_parameter_requires_grad(model_ft)
        elif "mlp1_utk" in self.model_name.lower():
            model_ft = MLP1(120000,self.num_classes)
            set_parameter_requires_grad(model_ft)
        elif "conv2_cifar" in self.model_name.lower():
            model_ft = Conv2_CIFAR(self.num_classes)
            set_parameter_requires_grad(model_ft)
        else:
            # feature_extract = False #finetune the whole network when False
            model_ft, input_size = initialize_model(self.model_name.lower(),
                                                    self.num_classes, 
                                                    feature_extract=None,
                                                    use_pretrained=True)
            set_parameter_requires_grad(model_ft)
        # else:
        #     raise ValueError("Model name {} not identified!".format(self.model_name))
        return model_ft


# In[ ]:


def train_model(model, num_epochs, device, data_loaders, criterion_kwargs={}, checkpoint=None):
    
    """
    For model training.
    
    model: instance of DNN
    num_epochs: epochs for which the model must be trained
    device: GPU
    data_loaders: defined in experiments.ipynb; allows for iteration over the dataset in a principled way
    criterion_kwargs: for usual training this is an empty dict. 
        When training with regularization, this is a dictionary with keys 'protected_classes' and 
        'inputs' with None values. 
        A wrapper function sets these values for each batch.
    """
    
    if checkpoint is not None:
        print ('Resuming training from {} epoch!'.format(checkpoint))
        filename = '{}_epoch_{}_lr_{}.pth'.format(model.model_name, checkpoint, learning_rate)
        model.model_ft.load_state_dict(torch.load('../{}/model_weights/{}'.format(ds_obj.name, filename),
                                                 map_location=device))
    model.model_ft.train()
    # we need model to be in train mode regardless since regularized loss requires calculation of gradients
    model.model_ft = model.model_ft.to(device)
    model.model_ft = model.model_ft.float()
    params_to_update = model.model_ft.parameters()
    optimizer = optim.SGD(params_to_update, lr=model.learning_rate, momentum=0.9)
    
    aggregate_coeff = model.aggregate_coeff # aggregate all test stats after every 5 epochs
    train_acc_history, train_total_loss_history, train_ce_loss_history, train_reg_history = [], [], [], []
    test_acc_history, test_total_loss_history, test_ce_loss_history, test_reg_history = [], [], [], []
    train_minority_dist, train_majority_dist, test_minority_dist, test_majority_dist = [], [], [], []
    best_acc = None
    since = time.time()
    
    iterator = range(num_epochs) if checkpoint is None else range(checkpoint + 1, num_epochs)
    
    for epoch in iterator:
        print('\nEpoch {}/{}\n'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for phase in ['train', 'test']:
            print (phase)
            running_loss, running_corrects = 0.0, 0.0
            
            epoch_predicted, epoch_true = None, None
            # Iterate over data.
            for _, inputs, labels, protected_classes in data_loaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                protected_classes = protected_classes.to(device)
                inputs.requires_grad = True
                if not isinstance(model.criterion, RegularizedLoss):
                    model.criterion = model.criterion.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

#                 with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                outputs = model.model_ft(inputs.float())
                loss = model.criterion(outputs, labels, 
                                       **hp.prepare_kwargs(criterion_kwargs, inputs, protected_classes, phase))
                _, preds = torch.max(outputs, 1)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                
#                 print ("Did backprop, going to sleep")
#                 time.sleep(5)
                
                # remove everything from GPU
                loss = loss.detach().cpu().numpy()
                preds = preds.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()
                inputs = inputs.detach().cpu().numpy()
                
#                 print ("removed everything from GPU, going to sleep")
#                 time.sleep(5)
                
                epoch_predicted = preds if epoch_predicted is None else np.concatenate((epoch_predicted, preds))
                epoch_true = labels if epoch_true is None else np.concatenate((epoch_true, labels))
                
#                 print ("Aggregated predicted, going to sleep")
#                 time.sleep(10)
                
                # statistics
                running_loss += loss * inputs.shape[0] # by default loss is normalized by number of samples
                running_corrects += np.sum(preds == labels)
                
#                 print ("Aggregated losses, going to sleep")
#                 time.sleep(10)
                
                torch.cuda.empty_cache()
#                 print ("Emptied cache, going to sleep")
#                 time.sleep(5)
#                 break
            
            if isinstance(model.criterion, RegularizedLoss):
                model.criterion.aggregate_stats(phase, verbose=True)
            
            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            epoch_acc = running_corrects / len(data_loaders[phase].dataset)

            if phase == 'train':
                if best_acc is None or best_acc < epoch_acc:
                    best_acc = epoch_acc
                    best_epoch = epoch
                    print ("For chosen weights, train accuracy: {}".format(epoch_acc))
                    print (hp.get_class_wise_accuracy(epoch_predicted, epoch_true, 
                                                   data_loaders[phase].dataset.ds.classes))
                    best_model_wts = copy.deepcopy(model.model_ft.state_dict())

            if epoch % aggregate_coeff == 0:
                if phase == 'test':
                    ## function defined in helper.ipynb
                    hp.persist_model_weights(model, data_loaders[phase].dataset.ds, model.learning_rate, epoch)

                # no validation yet; TODO: do Cross-Validation
                
                if phase == 'train':
                    train_acc_history.append(epoch_acc)
                    train_total_loss_history.append(epoch_loss)
                    if isinstance(model.criterion, RegularizedLoss):
                        train_ce_loss_history.append(model.criterion.cross_entropy_losses_epoch_train[-1])
                        train_reg_history.append(model.criterion.regularization_terms_epoch_train[-1])
                        train_majority_dist.append(model.criterion.d_approx_majority_epoch_train[-1])
                        train_minority_dist.append(model.criterion.d_approx_minority_epoch_train[-1])
                else:
                    test_acc_history.append(epoch_acc)
                    test_total_loss_history.append(epoch_loss)
                    if isinstance(model.criterion, RegularizedLoss):
                        test_ce_loss_history.append(model.criterion.cross_entropy_losses_epoch_test[-1])
                        test_reg_history.append(model.criterion.regularization_terms_epoch_test[-1])
                        test_majority_dist.append(model.criterion.d_approx_majority_epoch_test[-1])
                        test_minority_dist.append(model.criterion.d_approx_minority_epoch_test[-1])
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    model.model_ft.load_state_dict(best_model_wts)
    
    return (train_acc_history, train_total_loss_history, train_ce_loss_history, 
            train_reg_history, train_minority_dist, train_majority_dist,
            test_acc_history, test_total_loss_history, test_ce_loss_history, 
            test_reg_history, test_minority_dist, test_majority_dist)


# In[ ]:


def load_model_history(model, ds_obj, num_epochs, portion, device, override_criterion=None, criterion_kwargs={}):
    """
    
    """
    datasets = {x : PytorchLoader(ds=ds_obj, portion=x) for x in PHASES}
    data_loaders = {"{}".format(x) : torch.utils.data.DataLoader(datasets[x], 
                                                             batch_size=200, shuffle=False, 
                                                             num_workers=1) for x in PHASES}
    
    accuracy_history, accuracy_history_s0, accuracy_history_s1, losses = [], [], [], []
    for epoch in np.arange(0, num_epochs, model.aggregate_coeff):
        weights_filename = '../{}/model_weights/{}_epoch_{}_lr_{}.pth'.format(ds_obj.name, 
                                                                              model.model_name, 
                                                                              epoch,
                                                                              model.learning_rate)
        model.model_ft.load_state_dict(torch.load(weights_filename, map_location=device))
        model.model_ft = model.model_ft.to(device)
        
        true_classes, predicted_classes, protected_classes, loss = get_model_predictions(model, data_loaders, 
                                                                                         portion, device,
                                                                                         override_criterion,
                                                                                         criterion_kwargs)
        accuracy_history.append(
            accuracy_score(true_classes, predicted_classes))
        accuracy_history_s0.append(
            accuracy_score(true_classes[protected_classes == 0], predicted_classes[protected_classes == 0]))
        accuracy_history_s1.append(
            accuracy_score(true_classes[protected_classes == 1], predicted_classes[protected_classes == 1]))
        losses.append(loss)
        
        print ("Epoch: {}".format(epoch))
        
    return accuracy_history, accuracy_history_s0, accuracy_history_s1, losses


# In[ ]:


def get_model_predictions(model, data_loaders, portion, device, override_criterion=None, criterion_kwargs={}):
    """
    Given a model and data_loader, iterates over the data and returns the true_classes, predicted_classes
    and protect_class labels (in case of Adience for example, this is either male or female)
    Additionally outputs the average loss on that dataset
    """
    true_classes, predicted_classes, protected_classes = None, None, None
    running_loss = torch.tensor(0.0, device=device)
    for _, inputs, class_labels, protected_class in data_loaders[portion]:
#         print (inputs.size())
        inputs = inputs.to(device)
        class_labels = class_labels.to(device)
        
        if isinstance(model.model_ft.fc3.weight.type(), torch.cuda.DoubleTensor):
            model.model_ft = model.model_ft.float()
        model_outs = model.model_ft(inputs.float())
        if override_criterion is None:
            running_loss += model.criterion(model_outs, class_labels, 
                **hp.prepare_kwargs(criterion_kwargs, inputs, protected_class, portion)) * inputs.size(0)
        else:
            running_loss += override_criterion(model_outs, class_labels, 
                **hp.prepare_kwargs(criterion_kwargs, inputs, protected_class, portion)) * inputs.size(0)
        _, preds = torch.max(model_outs, 1)

        predicted_classes = preds if predicted_classes is None else                 torch.cat((predicted_classes, preds))
        true_classes = class_labels if true_classes is None else                 torch.cat((true_classes, class_labels))
        protected_classes = protected_class if protected_classes is None else                 torch.cat((protected_classes, protected_class))

        inputs, class_labels, model_outs, preds = None, None, None, None
    
    running_loss = running_loss.double()/len(data_loaders[portion].dataset)
    return (true_classes.cpu().numpy(), predicted_classes.cpu().numpy(), 
            protected_classes.cpu().numpy(), running_loss)

