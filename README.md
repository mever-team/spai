# SPAI: Spectral AI-Generated Image Detector
__Official code repository for the CVPR2025 paper [Any-Resolution AI-Generated Image Detection by Spectral Learning](https://arxiv.org/abs/2411.19417).__

<div align="center";">

**Dimitrios Karageorgiou<sup>1,2</sup>, Symeon Papadopoulos<sup>1</sup>, Ioannis Kompatsiaris<sup>1</sup>, Efstratios Gavves<sup>2,3</sup>**

<sup>1</sup> Information Technologies Institute, CERTH, Greece  
<sup>2</sup> University of Amsterdam, The Netherlands  
<sup>3</sup> Archimedes/Athena RC, Greece

</div>

<p align="center">
    <img src="docs/overview.svg" alt="Paper Overview" />
</p>

**SPAI employs spectral learning to learn the spectral distribution of real 
images under a self-supervised setup. Then, using the spectral 
reconstruction similarity it detects AI-generated images as out-of-distribution 
samples of this learned model.**

### :newspaper: News

- 28/03/25: Code released.
- 27/02/25: Paper accepted on CVPR2025.

## :hammer: Installation

### Hardware requirements

The code originally targeted Nvidia L40S 48GB GPU, however many recent cuda-enabled GPUs should be
supported. Inference should be effortless performed with less than 8GB of GPU RAM. As training originally
targeted a 48GB GPU, a suitable GPU should be presented to reproduce the paper's setup
without further modifications of the code. 

### Required libraries
#### Anaconda
To train and evaluate SPAI an anaconda environment can be used for installing all the 
required dependencies as following:

```bash
conda create -n spai python=3.11
conda activate spai
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
pip install -r requirements.txt
```

#### UV
The project can also be managed with [uv](https://docs.astral.sh/uv/). To use uv you should download it (https://docs.astral.sh/uv/getting-started/installation/).

To setup the environment go to the root of the project and run the following commands:
```bash
uv sync
```
This will install the correct python version and all dependencies (specified in the `pyproject.toml`) to an environment (`.venv`).

To execute a python program with uv you run `uv run python <path to file>`. So lets say you have a python file called main.py you want to run. Then you would execute this with:
```bash
uv run python main.py
```

Furthermore, the installation of [Nvidia APEX](https://github.com/NVIDIA/apex) is required for training.  

### Weights Checkpoint

The trained weights checkpoint can be downloaded [here](https://drive.google.com/file/d/1vvXmZqs6TVJdj8iF1oJ4L_fcgdQrp_YI/view?usp=sharing) 
and should be placed under the `weights` directory, located under the project's root directory.

## :fire: Inference

To compute the predicted scores for a set of images, place them under a directory
and use the following command.

```bash
python -m spai --input <input_dir> --output <output_dir>
```

where:
- `input_dir`: is a directory where the input images are located,
- `output_dir`: is a directory where a csv file with the predictions will be written.

The `--input` option also accepts CSV files containing the paths of the images. The CSV
files of the evaluation set, included under the `data` directory, can be used as examples.
For downloading the images of these evaluation CSVs, check the instruction [here](docs/data.md).

## :triangular_ruler: Architecture Overview

<p align="center">
    <img src="docs/architecture.svg" alt="Overview of the SPAI architecture" />
</p>

We learn a model of the spectral distribution of real images under a self-supervised setup using
masked spectral learning. Then, we use the spectral reconstruction similarity to measure the divergence from this learned distribution and
detect AI-generated images as out-of-distribution samples of this model. Spectral context vector captures the spectral context under which
the spectral reconstruction similarity values are computed, while spectral context attention enables the processing of any-resolution images
for capturing subtle spectral inconsistencies.

## :muscle: Training

### Required pre-trained model
Download the pre-trained ViT-B/16 MFM model from its [public repo](https://github.com/Jiahao000/MFM)
and place it under the `weights` directory:
```txt
weights
|_ mfm_pretrain_vit_base.pth
```

### Required data
Latent diffusion training and validation data can be downloaded from their corresponding [repo](https://github.com/grip-unina/DMimageDetection).
Furthermore, the corresponding instructions for downloading COCO and LSUN should be followed. 
They should be placed under the `datasets` directory as following:
```txt
datasets
|_latent_diffusion_trainingset
  |_train
    ...
  |_val
    ...
|_COCO
  ...
|_LSUN
  ...
```

Then, a csv file describing these data should be created as following:

```bash
python spai/create_dmid_ldm_train_val_csv.py \
  --train_dir "./datasets/latent_diffusion_trainingset/train" \
  --val_dir "./datasets/latent_diffusion_trainingset/val" \
  --coco_dir "./datasets/COCO" \
  --lsun_dir "./datasets/LSUN" \
  -o "./datasets/ldm_train_val.csv"
```

The validation split can be augmented as following:

```bash
python spai/tools/augment_dataset.py \
  --cfg ./configs/vit_base/vit_base__multipatch__100ep__intermediate__restore__patch_proj_per_feature__last_proj_layer_no_activ__fre_orig_branch__all_layers__bce_loss__light_augmentations.yaml \
  -c ./datasets/ldm_val.csv \
  -o ./datasets/ldm_val_augm.csv \
  -d ./datasets/latent_diffusion_trainingset_augm
```

Then, training can be performed as following:

```bash
python -m spai train \
  --cfg "./configs/spai.yaml" \
  --batch-size 72 \
  --pretrained "./weights/mfm_pretrain_vit_base.pth" \
  --output "./output/train" \
  --data-path "./datasets/ldm_train_val.csv" \
  --tag "spai" \
  --amp-opt-level "O2" \
  --data-workers 8 \
  --save-all \
  --opt "DATA.VAL_BATCH_SIZE" "256" \
  --opt "MODEL.FEATURE_EXTRACTION_BATCH" "400" \
  --opt "DATA.TEST_PREFETCH_FACTOR" "1"
```

## :mag_right: Evaluation

When a model has been trained using the previous script, it can be evaluated as following:

```bash
python -m spai test \
  --cfg "./configs/spai.yaml" \
  --batch-size 8 \
  --model "./output/train/finetune/spai/<epoch_name>.pth" \
  --output "./output/spai/test" \
  --tag "spai" \
  --opt "MODEL.PATCH_VIT.MINIMUM_PATCHES" "4" \
  --opt "DATA.NUM_WORKERS" "8" \
  --opt "MODEL.FEATURE_EXTRACTION_BATCH" "400" \
  --opt "DATA.TEST_PREFETCH_FACTOR" "1" \
  --test-csv "<test_csv_path>"
```

where:
- `test_csv_path`: Path to a csv file including the paths of the testing data.
- `epoch_name`: Filename of the epoch selected during validation. 

## :star2: Acknowledgments

This work was partly supported by the Horizon Europe
projects [ELIAS](https://elias-ai.eu/) (grant no. 101120237) and [vera.ai](https://www.veraai.eu/home) (grant
no. 101070093). The computational resources were granted
with the support of [GRNET](https://grnet.gr/en/).

Pieces of code from the [MFM](https://github.com/Jiahao000/MFM) project 
have been used as a basis for developing this project. We thank its 
authors for their contribution.

## :black_nib: License & Contact

This project will download and install additional third-party open 
source software projects. Also, all the employed third-party data 
retain their original license. Review their license terms 
before use.  

The source code and model weights of this project are released under 
the [Apache 2 License](https://www.apache.org/licenses/LICENSE-2.0).

For any question you can contact [d.karageorgiou@uva.nl](mailto:d.karageorgiou@uva.nl). 

## :scroll: Citation

If you found this work useful for your research, you can cite the following paper:

```text
@article{karageorgiou2025any,
  title={Any-Resolution AI-Generated Image Detection by Spectral Learning},
  author={Karageorgiou, Dimitrios and Papadopoulos, Symeon and Kompatsiaris, Ioannis and Gavves, Efstratios},
  journal={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}
```
