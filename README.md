# DenoiseLoc

[DenoiseLoc: Boundary Denoising for Video Activity Localization](https://arxiv.org/abs/2304.02934), ICLR 2023

[Mengmeng Xu](https://mengmengxu.netlify.app/),
[Mattia Soldan](https://www.mattiasoldan.com/),
[Jialin Gao](https://scholar.google.com/citations?user=sj4FqEgAAAAJ&hl=zh-CN),
[Shuming Liu](https://scholar.google.ae/citations?user=gPcJ6YkAAAAJ&hl=en),
[Juan-Manuel Pérez-Rúa](https://scholar.google.com/citations?user=Vbvimu4AAAAJ&hl=es),
[Bernard Ghanem](https://www.bernardghanem.com/)

This repo host the original code of our DenoiseLoc work, along with a copy of QVHighlights dataset for moment retrieval and highlight detections.
DenoiseLoc is an encoder-decoder model to tackle the video activity localization problem from a denoising perspective. During training, a set of action spans is randomly generated from the ground truth with a controlled noise scale. The inference reverses this process by boundary denoising, allowing the localizer to predict activities with precise boundaries and resulting in faster convergence speed. This code works for QV-Highlights dataset, where we observe a gain of +12.36% average mAP over the baseline.

The code is developed on top of [Moment-DETR](https://github.com/jayleicn/moment_detr), we keep minimal changes for simplicity but make necessary adaption for clarity. We keep the official data and evaluation tools in folders *data* and *standalone_eval*, respectively.

![teaser](./teaser.jpg)


## Table of Contents

- [DenoiseLoc](#denoiseloc)
  - [Table of Contents](#table-of-contents)
  - [Prerequisites](#prerequisites)
  - [Training](#training)
  - [Inference](#inference)
  - [More](#more)
  - [Acknowledgement](#acknowledgement)
  - [LICENSE](#license)




## Prerequisites
0. Clone this repo

```
git clone https://github.com/frostinassiky/denoise_loc.git
cd moment_detr
```

1. Prepare feature files

Download [moment_detr_features.tar.gz](https://drive.google.com/file/d/1Hiln02F1NEpoW8-iPZurRyi-47-W2_B9/view?usp=sharing) (8GB),
extract it under project root directory:
```
tar -xf path/to/moment_detr_features.tar.gz
```
The features are extracted using Linjie's [HERO_Video_Feature_Extractor](https://github.com/linjieli222/HERO_Video_Feature_Extractor).

2. Install dependencies.

This code requires Python 3.7, PyTorch, and a few other Python libraries.
We recommend creating conda environment and installing all the dependencies as follows:
```
# create conda env
conda create --name denoise_loc python=3.7
# activate env
conda activate denoise_loc
# install pytorch with CUDA 11.6
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
# install other python packages
pip install tqdm ipython easydict tensorboard tabulate scikit-learn pandas
# compile and install nms1d and align1d
cd lib/align1d
python setup.py install
cd ../nms1d
python setup.py install
```
The PyTorch version we tested is `1.9.0`.

## Training

Training can be launched by running the following batch scrip:
```
sbatch slurm/trainval_v100_snr.sh
```

Alternatively, you may revise the original training script and run it locally:
```
bash moment_detr/scripts/train.sh
```
For more configurable options, please checkout the config file [moment_detr/config.py](moment_detr/config.py).

## Inference
Once the model is trained, you can use the following command for inference:
```
bash moment_detr/scripts/inference.sh CHECKPOINT_PATH SPLIT_NAME
```
where `CHECKPOINT_PATH` is the path to the saved checkpoint, `SPLIT_NAME` is the split name for inference, can be one of `val` and `test`.

## More
Since our project is developed from [Moment-DETR](https://github.com/jayleicn/moment_detr). Please refer to their codebase for:
1. Pretraining and Finetuning
2. Evaluation and Codalab Submission
3. Train Moment-DETR on your own dataset
4. Run predictions on your own videos and queries


## Acknowledgement
This code is based on [Moment-DETR](https://github.com/jayleicn/moment_detr), and the implementation is refered from [G-TAD](https://github.com/frostinassiky/gtad) and [DiffusionDet](https://github.com/ShoufaChen/DiffusionDet). We thank the authors for their awesome open-source contributions.

## LICENSE
The annotation files are under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license, see [./data/LICENSE](data/LICENSE). All the code are under [MIT](https://opensource.org/licenses/MIT) license, see [LICENSE](./LICENSE).