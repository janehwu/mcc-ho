# MCC-HO: Multiview Compressive Coding for Hand-Object 3D Reconstruction
Code repository for the paper:
**Reconstructing Hand-Held Objects in 3D from Images and Videos**

[Jane Wu](https://janehwu.github.io/), [Georgios Pavlakos](https://geopavlakos.github.io/), [Georgia Gkioxari](https://gkioxari.github.io/), [Jitendra Malik](http://people.eecs.berkeley.edu/~malik/)

[![arXiv](https://img.shields.io/badge/arXiv-2404.06507-00ff00.svg)](https://arxiv.org/pdf/2404.06507.pdf)  [![Website shields.io](https://img.shields.io/website-up-down-green-red/http/shields.io.svg)](https://janehwu.github.io/mcc-ho)

<p align="center">
<img width="1280" alt="teaser" src="https://janehwu.github.io/mcc-ho/mccho_results.png">
</p>

## Installation
Installation and preparation follow [the MAE repo](https://github.com/facebookresearch/mae), just like [MCC](https://github.com/facebookresearch/MCC).
Please also install [PyTorch3D](https://pytorch3d.org/) for 3D related funcionality and the following libraries:

```
pip install omegaconf trimesh
```

### HaMeR Installation
Our demo includes the option to use HaMeR to obtain 3D hands from RGB images. If you would like to use HaMeR, you can initialize the submodule:

```
git submodule update --init --recursive
```

Please follow [the HaMeR repo](https://github.com/geopavlakos/hamer) for installation instructions. Afer installing HaMeR, you will need to create a symlink to the HaMeR data/checkpoints to be used for our demo:

```
ln -s [path to HaMeR _DATA folder] .
```

## Data
Please see [DATASET.md](DATASET.md) for information on data preparation.

## Demo (with HaMeR)
If you do not have input 3D hands, you can run MCC-HO inference on any input image and use HaMeR to obtain 3D hands. Please use, e.g.,
```
python demo_with_hamer.py \
    --image demo/drink_v_1F96GArORtg_frame000084.jpg \
    --obj_seg demo/drink_v_1F96GArORtg_frame000084_mask.png \
    --cam demo/camera_intrinsics_hamer.json \
    --checkpoint [path to model checkpoint]
```

The object segmentation mask (`obj_seg`) can be obtained using any off-the-shelf segmentation model, e.g. we use [SAM 2](https://github.com/facebookresearch/sam2).

One may use a checkpoint from the training step below or download our pretrained model (trained on DexYCB, MOW, and HOI4D) [[here](https://drive.google.com/file/d/17VOYtywmKhDh_JUULT_M20TNByBUUbqZ/view?usp=sharing)]. One may set the `--score_thresholds` argument to specify the score thresholds (More points are shown with a lower threshold, but the predictions might be noisier).

The script will generate an html file showing an interactive visualizaion of the MCC-HO output with [plotly](https://plotly.com/), as well as the predicted point clouds in the same coordinate system as the preprocessed HaMeR hand (saved at `out_demo/input_hand.obj`).

## Demo (without HaMeR)
To run MCC-HO inference on any input image and input 3D hand, please use, e.g.,
```
python demo.py \
    --image demo/boardgame_v_W_qdSiPKSdQ_frame000019.jpg \
    --hand demo/boardgame_v_W_qdSiPKSdQ_frame000019_hand.obj \
    --seg demo/boardgame_v_W_qdSiPKSdQ_frame000019_mask.png \
    --cam demo/camera_intrinsics_mow.json \
    --checkpoint [path to model checkpoint]
```

**Note that for your own data, you need to make sure the camera intrinsics correspond to the input 3D hand.**

**[Same as above]** One may use a checkpoint from the training step below or download our pretrained model (trained on DexYCB, MOW, and HOI4D) [[here](https://drive.google.com/file/d/17VOYtywmKhDh_JUULT_M20TNByBUUbqZ/view?usp=sharing)]. One may set the `--score_thresholds` argument to specify the score thresholds (More points are shown with a lower threshold, but the predictions might be noisier).

The script will generate an html file showing an interactive visualizaion of the MCC-HO output with [plotly](https://plotly.com/), as well as the predicted point clouds in the same coordinate system as the input hand.

## Training
To train an MCC-HO model, please run
```
OUTPUT=model_outputs
python main_mccho.py \
    --mccho_path [path to MCC-HO preprocessed data] \
    --dataset_cache [path to dataset cache] \
    --job_dir $OUTPUT \
    --output_dir $OUTPUT/log \
    --shuffle_train
```
- Optional: MCC-HO (excluding the segmentation output layers) may be initialized using MCC pre-trained. A pretrained MCC model is available [[here](https://dl.fbaipublicfiles.com/MCC/co3dv2_all_categories.pth)].

## Acknowledgements
This implementation builds on the [MCC](https://github.com/facebookresearch/MCC) codebase, which in turn is based on [MAE](https://github.com/facebookresearch/mae).

## Citing
If you find this code useful for your research, please consider citing the following paper:

```bibtex
@article{wu2024reconstructing,
  title={Reconstructing Hand-Held Objects in 3D},
  author={Wu, Jane and Pavlakos, Georgios and Gkioxari, Georgia and Malik, Jitendra},
  journal={arXiv preprint arXiv:2404.06507,
  year={2024},
}
```
