# MCC-HO: Multiview Compressive Coding for Hand-Object 3D Reconstruction
Code repository for the paper:
**Reconstructing Hand-Held Objects in 3D**

[Jane Wu](https://janehwu.github.io/), [Georgios Pavlakos](https://geopavlakos.github.io/), [Georgia Gkioxari](https://gkioxari.github.io/), [Jitendra Malik](http://people.eecs.berkeley.edu/~malik/)

[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-00ff00.svg)](https://janehwu.github.io/mcc-ho/)  [![Website shields.io](https://img.shields.io/website-up-down-green-red/http/shields.io.svg)](https://janehwu.github.io/mcc-ho/)

<p align="center">
<img width="720" alt="teaser" src="https://github.com/janehwu/mcc-ho/assets/9442165/b8e61d2a-90cf-4b2c-a722-6555b83661a8">
</p>

## Installation
Installation and preparation follow [the MAE repo](https://github.com/facebookresearch/mae), following [MCC](https://github.com/facebookresearch/MCC).
Please also install [PyTorch3D](https://pytorch3d.org/) for 3D related funcionality and the following libraries:

```
pip install omegaconf trimesh
```

## Data
[TBD] Please see [DATASET.md](DATASET.md) for information on data preparation.

## Demo
To run MCC-HO inference on any input image and estimated 3D hand, please use, e.g., 
```
python demo.py \
    --image demo/boardgame_v_W_qdSiPKSdQ_frame000019.jpg \
    --hand demo/boardgame_v_W_qdSiPKSdQ_frame000019_hand.obj \
    --seg demo/boardgame_v_W_qdSiPKSdQ_frame000019_mask.png \
    --cam demo/camera_intrinsics_mow.json \
    --checkpoint [path to model checkpoint]
```
One may use a checkpoint from the training step below or download our pretrained model (trained on DexYCB, MOW, and HOI4D) [[here](https://drive.google.com/file/d/17VOYtywmKhDh_JUULT_M20TNByBUUbqZ/view?usp=sharing)].
One may set the `--score_thresholds` argument to specify the score thresholds (More points are shown with a lower threshold, but the predictions might be noisier). 
The script will generate an html file showing an interactive visualizaion of the MCC-HO output with [plotly](https://plotly.com/).

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
@inproceedings{wu2024reconstructing,
  title={Reconstructing Hand-Held Objects in 3D},
  author={Wu, Jane and Pavlakos, Georgios and Gkioxari, Georgia and Malik, Jitendra},
  journal={arXiv preprint arXiv:XXXX},
  year={2024},
}
```
