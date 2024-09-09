# [Feedback-guided Data Synthesis for Imbalanced Classification (TMLR 2024 accepted paper)](https://openreview.net/forum?id=IHJ5OohGwr)

[![arXiv](https://img.shields.io/badge/arXiv-2310.00158-b31b1b.svg)](https://arxiv.org/abs/2310.00158)

> **Feedback-guided Data Synthesis for Imbalanced Classification**<br>
> Reyhane Askari Hemmat<sup>1, 2, 3</sup>, Mohammad Pezeshki *<sup>1</sup>, Florian Bordes<sup>1</sup>, Michal Drozdzal<sup>1</sup>, Adriana Romero-Soriano<sup>1, 2, 3, 4</sup>
> <sup>1</sup> FAIR at Meta, <sup>2</sup> Mila,<sup>3</sup> Universite de Montreal, <sup>4</sup> McGill University, <sup>5</sup> Canada CIFAR AI chair
>**Abstract**: <br>
Current status quo in machine learning is to use static datasets of real images for training, which often come from long-tailed distributions. With the recent advances in generative models, researchers have started augmenting these static datasets with synthetic data, reporting moderate performance improvements on classification tasks. We hypothesize that these performance gains are limited by the lack of feedback from the classifier to the generative model, which would promote the usefulness of the generated samples to improve the classifier’s performance. In this work, we introduce a framework for augmenting static datasets with useful synthetic samples, which leverages one-shot feedback from the classifier to drive the sampling of the generative model. In order for the framework to be effective, we find that the samples must be close to the support of the real data of the task at hand, and be sufficiently diverse. We validate three feedback criteria on a long-tailed dataset (ImageNet-LT) as well as a group-imbalanced dataset (NICO++). On ImageNet-LT, we achieve state-of-the-art results, with over 4% improvement on underrepresented classes while being twice efficient in terms of the number of generated synthetic samples. NICO++ also enjoys marked boosts of over 5% in worst group accuracy. With these results, our framework paves the path towards effectively leveraging state-of-the-art text-to-image models as data sources that can be queried to improve downstream applications.
## Description
This repo contains the official code for feedback guided data synthesis. We provide the script for image generation with feedback guidance. For classification, we used the codebase of https://github.com/facebookresearch/classifier-balancing. 
## Usage
This codebase generates samples of an LDM with feedback guidance from a pretrained classifier and saves the generations in a systematic way. We assume the classifier is already pre-trained on the real data. In this code for ease of use, we use a pretrained ResNet model but in the paper the classifier is only trained on long-tailed data.
### Creating csv files of generations
As a first step, we define all the hyper-parameters of generations in the `generate_csvs_offline.py` and run this to create the csv. Make sure to specify the correct `output_dir`.
This script populates a csv file corresponding to the defined set of hyper-parameters.
### Generation
Given an `output_dir` from the previous step, the generation code reads the csv file in that directory and generates the samples. The code supports generating on multiple gpus in parallel (use `--submit_it`). There is no need for the gpus to be on the same node.
```
python generate.py --output_dir $OUTPUTDIR
```
The samples are saved in the '$OUTPUTDIR'. This generation code runs a custom pipe for a predefined latent diffusion model. We implement feedback guidance in the `fg_pipe.py` file in the `cond_fn` function.
## Tips and Tricks
- One can use this code with any pre-trained latent diffusion model. You will need to fix some imports based on your preference of the LDM model selected from the hugging face library. Specifically, in the `utils_gen.py` script, you need to define `CLIP_MODEL_PATH` and `LDM_MODEL_PATH` variables. Note that the code released here is just an example of how feedback guidance can be applied to a large scale LDM. 

## License 
This code is licensed under CC-BY-NC license.


## Citation

If you make use of our work or code, please cite our paper:
```
@article{hemmat2024feedback,
  title={Feedback-guided data synthesis for imbalanced classification},
  author={Askari-Hemmat, Reyhane and Pezeshki, Mohammad and Bordes, Florian and Drozdzal, Michal and Romero-Soriano, Adriana},
  journal={arXiv preprint arXiv:2310.00158},
  year={2023}
}
```
