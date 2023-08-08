# Adaptive IMLE for Few-shot Pretraining-free Generative Modelling

Official PyTorch implementation of the ICML 2023 paper

<!-- include image from ./assets/img.png -->

![img](./assets/res0.png)
![img](./assets/res1.png)

### Abstract

_Despite their success on large datasets, GANs have been difficult to apply in the few-shot setting, where only a limited number of training examples are provided. Due to mode collapse, GANs tend to ignore some training examples, causing overfitting to a subset of the training dataset, which is small in the first place. A recent method called Implicit Maximum Likelihood Estimation (IMLE) is an alternative to GAN that tries to address this issue. It uses the same kind of generators as GANs but trains it with a different objective that encourages mode coverage. However, the theoretical guarantees of IMLE hold under a restrictive condition that the optimal likelihood at all data points is the same. In this paper, we present a more generalized formulation of IMLE which includes the original formulation as a special case, and we prove that the theoretical guarantees hold under weaker conditions. Using this generalized formulation, we further derive a new algorithm, which we dub Adaptive IMLE, which can adapt to the varying difficulty of different training examples. We demonstrate on multiple few-shot image synthesis datasets that our method significantly outperforms existing methods._

[Paper PDF](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=sKWTHpsAAAAJ&citation_for_view=sKWTHpsAAAAJ:u5HHmVD_uO8C).

## Requirements

Python3.8

```bash
virtualenv -p python3.8 venv
pip3 install -r requirements.txt
pip3 install dciknn_cuda-0.1.15.tar.gz
```

## Pretrained Models

Pretrained models can be downloaded from the following link:
https://drive.google.com/file/d/1X8nl1TWjv2w_zk_8FoRhtht0au3jvI8k/view?usp=sharing

## Datasets

Running the following will doownload and extract all datasets used in this project.

```bash
bash ./scripts/setup_datasets.sh
```

Alternatively, you can download the datasets manually from the following links:
https://drive.google.com/file/d/1VwFFzU8wJD1XJtfg60iLwnyBQ_cLZObL/view?usp=drive_link

## Reproducing Results

The reported results can be reproduced using the scripts in the `reproduce` folder. For example, to reproduce the results on the `FFHQ` dataset, run the following command:

```bash
python ./reproduce/ffhq.sh
```

Make sure to set the `--data_root` flag to the path where the datasets are stored.

## Training on Custom Datasets

Change the `--data_root` flag to the path where the datasets are stored. Also, see the [Important Hyperparameters](#important-hyperparameters) section for appropriate set of hyperparameters to be used.

## Important Hyperparameters

`--data_root`:
path to the image folder dataset

`--chagne_coef`: `\tau` in the paper, the percentage of threshold before considering the subproblem as solved.

`--force_factor`: defines the number of random samples to be generated for the nearest neighbour finding part in terms of dataset length. E.g., `2` means 2 \* _dataset_length_ random samples will be generated. We have kept the number of random samples around 10k.

`--lr`: learning rate.

## Notebook

A very simple (and not efficient) implementation of IMLE and AdaptiveIMLE along with a toy training example can be found in the `notebooks` folder.

## Wandb

If you set the following wandb parameters the metrics including loss and FID score throughout the training will be logged to wandb.

`--use_wandb`: set to `True` to log to wandb.

`--wandb_project`: wandb project name.

`--wandb_name`: run name.

## Citation
```@inproceedings{aghabozorgi2023adaimle,
title={Adaptive IMLE for Few-shot Pretraining-free Generative Modelling
},
author={Mehran Aghabozorgi and Shichong Peng and Ke Li},
booktitle={International Conference on Machine Learning},
year={2023}
}```