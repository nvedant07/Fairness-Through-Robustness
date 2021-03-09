# evaluate a smoothed classifier on a dataset
import torch.nn as nn
from torchvision import datasets, models, transforms

import argparse
import datetime
import os
from time import time

from model_schema2 import MLP1, DeepCNN, UTKClassifier
# from architectures import get_architecture
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
    ORDERED_CLASS_LABELS = ['0-15', '15-25', '25-40', '40-60', '60+']
    # checkpoint = torch.load(args.base_classifier, map_location=device)
    # base_classifier = get_architecture(checkpoint["arch"], args.dataset)
    # base_classifier.load_state_dict(checkpoint['state_dict'])
    
    if 'mlp' in args.model_name.lower():
        base_classifier = MLP1(120000, len(ORDERED_CLASS_LABELS))
    elif 'utk_classifier' in args.model_name.lower():
        base_classifier = UTKClassifier(len(ORDERED_CLASS_LABELS))
    elif 'resnet' in args.model_name.lower():
        base_classifier = models.resnet18(pretrained=True)
        num_ftrs = base_classifier.fc.in_features
        base_classifier.fc = nn.Linear(num_ftrs, len(ORDERED_CLASS_LABELS))
    elif 'squeezenet' in args.model_name.lower():
        base_classifier = models.squeezenet1_0(pretrained=True)
        base_classifier.classifier[1] = nn.Conv2d(512, len(ORDERED_CLASS_LABELS), kernel_size=(1,1), stride=(1,1))
        base_classifier.num_classes = len(ORDERED_CLASS_LABELS)
    elif 'alexnet' in args.model_name.lower():
        base_classifier = models.alexnet(pretrained=True)
        num_ftrs = base_classifier.classifier[-1].in_features
        base_classifier.classifier[-1] = nn.Linear(num_ftrs, len(ORDERED_CLASS_LABELS))
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
    if not os.path.exists(args.dataset):
        os.mkdir(args.dataset)
    f = open('{}/{}'.format(args.dataset, args.outfile), 'w')
    print("idx\tlabel\tgender\trace\tbase_prediction\tsmooth_prediction\tradius\ttime", file=f, flush=True)

    # iterate through the dataset
    dataset = get_dataset(args.dataset, args.split)
    for i in range(len(dataset)):

        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break

        (img_idx, x, label) = dataset[i]
        if args.split == 'train':
	        gender_idx, race_idx = dataset.ds.train_image_genders[img_idx], dataset.ds.train_image_races[img_idx]
        else:
            gender_idx, race_idx = dataset.ds.test_image_genders[img_idx], dataset.ds.test_image_races[img_idx]

        before_time = time()
        # certify the prediction of g around x
        x = x.cuda()
        x = x.to(device)

        base_logits = base_classifier(x.unsqueeze(0))
        base_prediction = int(torch.argmax(base_logits, dim=1))

        smooth_prediction, radius = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.batch)
        after_time = time()

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        print("{}\t{}\t{}\t{}\t{}\t{}\t{:.3}\t{}".format(
            i, label, dataset.ds.gender_id_to_label[gender_idx], dataset.ds.race_id_to_label[race_idx], base_prediction, 
          	smooth_prediction, radius, time_elapsed), file=f, flush=True)

    f.close()
