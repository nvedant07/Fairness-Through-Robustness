# evaluate a smoothed classifier on a dataset
import torch.nn as nn
from torchvision import datasets, models, transforms

import argparse
import datetime
import os
from time import time

from model_schema2 import MLP1, DeepCNN, DeepCNNCifar100, make_layers
# from architectures import get_architecture
from pyramidnet import PyramidNet
from core import Smooth
from datasets import get_dataset, DATASETS, get_num_classes
import torch


parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("model_name", type=str, help="name of model to be loaded")
parser.add_argument("sigma", type=float, help="noise hyperparameter")
parser.add_argument("outfile", type=str, help="output file")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument("--gpu", type=int, default=0, help="GPU number")
args = parser.parse_args()

if __name__ == "__main__":
    # load the base classifier
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    if args.dataset.lower() == 'cifar10':
        ORDERED_CLASS_LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    elif args.dataset.lower() == 'cifar100':
        ORDERED_CLASS_LABELS = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
            'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
            'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
            'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
            'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
            'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
            'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
            'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
            'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
            'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
            'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
            'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank',
            'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
            'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
    elif args.dataset.lower() == 'cifar100super':
        ORDERED_CLASS_LABELS = ['aquatic_mammals', 'fish', 'flowers', 'food_containers',
            'fruit_and_vegetables', 'household_electrical_devices',
            'household_furniture', 'insects', 'large_carnivores',
            'large_man-made_outdoor_things', 'large_natural_outdoor_scenes',
            'large_omnivores_and_herbivores', 'medium_mammals',
            'non-insect_invertebrates', 'people', 'reptiles', 'small_mammals',
            'trees', 'vehicles_1', 'vehicles_2']
            
    # checkpoint = torch.load(args.base_classifier, map_location=device)
    # base_classifier = get_architecture(checkpoint["arch"], args.dataset)
    # base_classifier.load_state_dict(checkpoint['state_dict'])
    
    if 'mlp' in args.model_name.lower():
        base_classifier = MLP1(3072, len(ORDERED_CLASS_LABELS))
    elif 'pyramidnet' in args.model_name.lower():
        assert '_alpha_' in args.model_name.lower() and '_depth_' in args.model_name.lower(), \
            'alpha and depth must be passed in model name'
        _, alpha = args.model_name.lower().split('_alpha_')
        alpha = int(alpha.split('_')[0])
        _, depth = args.model_name.lower().split('_depth_')
        depth = int(depth.split('_')[0])
        kwargs = {'depth': depth, 'alpha': alpha, 'bottleneck': True if 'bottleneck' in args.model_name.lower() else False}
        base_classifier = PyramidNet(len(ORDERED_CLASS_LABELS), **kwargs)
    elif 'deep_cnn' in args.model_name.lower():
        base_classifier = DeepCNN(len(ORDERED_CLASS_LABELS))
    elif 'deep_cnn_cifar100' in args.model_name.lower():
        n_channel = 128
        cfg = [n_channel, n_channel, 'M', 2*n_channel, 2*n_channel, 'M', 
               4*n_channel, 4*n_channel, 'M', (8*n_channel, 0), 'M']
        layers = make_layers(cfg, batch_norm=True)
        model_ft = DeepCNNCifar100(layers, n_channel=8*n_channel, num_classes=len(ORDERED_CLASS_LABELS))
    elif 'resnet' in args.model_name.lower():
        base_classifier = models.resnet18(pretrained=True)
        num_ftrs = base_classifier.fc.in_features
        base_classifier.fc = nn.Linear(num_ftrs, len(ORDERED_CLASS_LABELS))
    elif 'squeezenet' in args.model_name.lower():
        base_classifier = models.squeezenet1_0(pretrained=True)
        base_classifier.classifier[1] = nn.Conv2d(512, len(ORDERED_CLASS_LABELS), kernel_size=(1,1), stride=(1,1))
        base_classifier.num_classes = len(ORDERED_CLASS_LABELS)
    elif 'densenet' in args.model_name.lower():
        base_classifier = models.densenet121(pretrained=True)
        num_ftrs = base_classifier.classifier.in_features
        base_classifier.classifier = nn.Linear(num_ftrs, len(ORDERED_CLASS_LABELS))
    elif 'vgg' in args.model_name.lower():
        base_classifier = models.vgg19_bn(pretrained=True)
        num_ftrs = base_classifier.classifier[6].in_features
        base_classifier.classifier[6] = nn.Linear(num_ftrs, len(ORDERED_CLASS_LABELS))
    else:
        raise ValueError("{} not a valid model!".format(args.model_name))
        
    base_classifier.load_state_dict(torch.load(args.base_classifier, map_location=device))
    base_classifier.eval()
    base_classifier = base_classifier.to(device)

    # create the smooothed classifier g
    smoothed_classifier = Smooth(base_classifier, get_num_classes(args.dataset), args.sigma, device=device)

    # prepare output file
    if not os.path.exists(args.dataset.lower()):
        os.mkdir(args.dataset.lower())
        
    f = open('{}/{}'.format(args.dataset.lower(), args.outfile), 'w')
    print("idx\tlabel\tbase\tsmooth\tradius\ttime", file=f, flush=True)

    # iterate through the dataset
    dataset = get_dataset(args.dataset, args.split)
    for i in range(len(dataset)):

        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break
        
        if 'cifar100super' in args.dataset:
            (idx, x, label) = dataset[i]
            assert idx == i
        else:
            (x, label) = dataset[i]

        before_time = time()
        # certify the prediction of g around x
        x = x.cuda()
        x = x.to(device)

        base_logits = base_classifier(x.unsqueeze(0))
        base_prediction = int(torch.argmax(base_logits, dim=1))

        smooth_prediction, radius = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.batch)
        after_time = time()

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        print("{}\t{}\t{}\t{}\t{:.3}\t{}".format(
            i, label, base_prediction, smooth_prediction, radius, time_elapsed), file=f, flush=True)

    f.close()
