# evaluate a smoothed classifier on a dataset
import argparse
import datetime
import os
from time import time

from model_schema2 import MLP1, DeepCNN
# from architectures import get_architecture
from core import Smooth
from datasets import get_dataset, DATASETS, get_num_classes
import torch


parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("sigma", type=float, help="noise hyperparameter")
parser.add_argument("outfile", type=str, help="output file")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
args = parser.parse_args()

if __name__ == "__main__":
    # load the base classifier
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ORDERED_CLASS_LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    #base_classifier = MLP1(len(ORDERED_CLASS_LABELS))
    base_classifier = DeepCNN(len(ORDERED_CLASS_LABELS))

    base_classifier.load_state_dict(torch.load(args.base_classifier, map_location=device))
    base_classifier.eval()
    base_classifier = base_classifier.to(device)

    # iterate through the dataset
    dataset = get_dataset(args.dataset, args.split)
    total = 0
    for i in range(len(dataset)):
        (x, label) = dataset[i]

        if isinstance(base_classifier, MLP1):
            # for MLP, flatten the input
            x = x.view(x.shape[0], np.product(x.shape[1:]))

        before_time = time()
        # certify the prediction of g around x
        x = x.cuda()

        logits = base_classifier(x.unsqueeze(0))
        prediction = torch.argmax(logits, dim=1)
        correct = int(prediction == label)

        total += correct
print(total)
