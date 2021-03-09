## Fairness Through Robustness: Investigating Robustness Disparity in Deep Learning

Official code to replicate results in [Fairness Through Robustness: Investigating Robustness Disparity in Deep Learning](https://arxiv.org/abs/2006.12621), presented at [FAccT 2021](https://facctconference.org/2021/acceptedpapers.html) in ~~Toronto, Canada~~ over Zoom.

Authors:

 * [Vedant Nanda](http://nvedant07.github.io/)
 * [Samuel Dooley](https://www.cs.umd.edu/~sdooley1/)
 * [Sahil Singla](https://scholar.google.co.in/citations?user=jjjbOI4AAAAJ&hl=en)
 * [Soheil Feizi](https://www.cs.umd.edu/~sfeizi/)
 * [John P. Dickerson](http://jpdickerson.com/)

Adversarial attacks in our code use [`foolbox`](https://github.com/bethgelab/foolbox). Code under `./smoothing_code` is taken from https://github.com/locuslab/smoothing and https://github.com/Hadisalman/smoothing-adversarial. We thank the authors of these repos for making their code public and easy to use.


### Requirements

 * python 3.7.x or higher
 * CUDA 10.1

python dependencies

 * To install all dependencies run `pip install -r requirements.txt`
 * NOTE: Our implementation uses foolbox v2.4.0. If not available on PyPi, we recommend building it from source. Follow instructions on the [Foolbox repo](https://github.com/bethgelab/foolbox) to build from source.

### Datasets

We use CIFAR-10, CIFAR-100, UTKFace and Adience for our analysis. See `./data` for more information on how to download the datasets.

### Training Models

We've released pre-trained weights used in our analysis (coming soon!). You can either use these weights or train models on your own.

#### Working with pre-trained weights

> Coming soon!

#### Training models

See `./code/experiment-main.py`.

### Evaluating Robustness Bias

This step requires running adversarial attacks (in our paper we evaluate [DeepFool](https://openaccess.thecvf.com/content_cvpr_2016/html/Moosavi-Dezfooli_DeepFool_A_Simple_CVPR_2016_paper.html) and [Carlini&Wagner](https://ieeexplore.ieee.org/abstract/document/7958570/)). 

#### Running Adversarial Attacks

See `./code/experiment-adversarial.py`

#### Running Randomized Smoothing 

```
cd smoothing_code
python certify.py cifar10 <path to model weights> <model_name> 0.125 <output_filename> --batch 100 --alpha 0.001 --N0 100 --N 100000 --skip 5 --gpu <gpu number>
```

#### Analysis

See `./code/certificate_analysis/` for analysis of randomized smoothing and `./code/experiment_adversarial_only_plot.ipynb` for analysis of adversarial attacks.

### Citation

If you found our work useful, please cite it. 

```
@inproceedings{nanda2021fairness,
    author = {Nanda, Vedant and Dooley, Samuel and Singla, Sahil and Feizi, Soheil and Dickerson, John P.},
    title = {Fairness Through Robustness: Investigating Robustness Disparity in Deep Learning},
    year = {2021},
    isbn = {9781450383097},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3442188.3445910},
    doi = {10.1145/3442188.3445910},
    booktitle = {Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency},
    pages = {466–477},
    numpages = {12},
    location = {Virtual Event, Canada},
    series = {FAccT '21}
}
```

### Feedback

We'd love to hear from you! Write to us at `vedant@cs.umd.edu` or `sdooley1@cs.umd.edu`.