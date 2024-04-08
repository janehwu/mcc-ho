# [TBD] Dataset Preparation

We use the dataset provider in [Implicitron](https://github.com/facebookresearch/pytorch3d/tree/main/pytorch3d/implicitron) for data loading. To speed up the loading, we cache the loaded meta data. Please run 
```
cd scripts
python prepare_data.py --dataset_path [path to dataset]
```
to generate the cache.

