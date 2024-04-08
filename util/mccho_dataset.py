# This source code is adapted from:
# MCC: https://github.com/facebookresearch/MCC
import random
from typing import cast

import torch
from pytorch3d.implicitron.dataset.dataset_base import FrameData

import util.dataset_utils as dataset_utils


def mccho_collate_fn(batch):
    assert len(batch[0]) == 5
    return (
        FrameData.collate([x[0] for x in batch]),
        FrameData.collate([x[1] for x in batch]),
        [x[2] for x in batch],
        [x[3] for x in batch],
        FrameData.collate([x[4] for x in batch]),
    )


def pad_point_cloud(pc, N):
    cur_N = pc._points_list[0].shape[0]
    if cur_N == N:
        return pc

    assert cur_N > 0

    n_pad = N - cur_N
    indices = random.choices(list(range(cur_N)), k=n_pad)
    pc._features_list[0] = torch.cat([pc._features_list[0], pc._features_list[0][indices]], dim=0)
    pc._points_list[0] = torch.cat([pc._points_list[0], pc._points_list[0][indices]], dim=0)
    return pc


# Custom functions for mccho
def combine_point_clouds(pc1, pc2):
    combined = pc1.detach().clone()
    combined._features_list[0] = torch.cat([pc1._features_list[0], pc2._features_list[0]], dim=0)
    combined._points_list[0] = torch.cat([pc1._points_list[0], pc2._points_list[0]], dim=0)
    return combined


def normalize(seen_xyz, mean=None, sd=None, sd_scale=3):
    if mean is None:
        mean = seen_xyz[torch.isfinite(seen_xyz.sum(dim=-1))].mean(axis=0)
    if sd is None:
        # Scale SD to account for the fact that only the hand is seen
        sd = seen_xyz[torch.isfinite(seen_xyz.sum(dim=-1))].std(dim=0).mean() * sd_scale
    seen_xyz = (seen_xyz - mean) / sd
    return seen_xyz, mean, sd


class MCCHODataset(torch.utils.data.Dataset):
    def __init__(self, args, is_train, is_viz=False, dataset_maps=None):

        self.args = args
        self.is_train = is_train
        self.is_viz = is_viz

        if is_train:
            self.dataset_split = 'train'
        elif self.args.run_viz:
            self.dataset_split = 'test'
        else:
            self.dataset_split = 'val'
        # Select dataset split
        if is_train:
            self.all_datasets = dataset_maps[0]
        elif self.args.run_viz:
            self.all_datasets = dataset_maps[2]
        else:
            self.all_datasets = dataset_maps[1]
        print(len(self.all_datasets), 'categories loaded')

        self.all_example_names = self.get_all_example_names()
        print('containing', len(self.all_example_names), 'examples')

    def get_all_example_names(self):
        all_example_names = []
        self.seen_frames = {}  # For test evaluation
        for category in self.all_datasets.keys():
            for sequence_name in self.all_datasets[category].seq_name2idx.keys():
                all_example_names.append((category, sequence_name))
                self.seen_frames[sequence_name] = 0
        return all_example_names

    def __getitem__(self, index):
        full_point_cloud = None
        for retry in range(1):
            try:
                if retry > 1:
                    index = random.choice(range(len(self)))
                    print('retry', retry, 'new index:', index)
                gap = 1 if (self.is_train or self.is_viz) else len(self.all_example_names) // len(self)
                assert gap >= 1
                category, sequence_name = self.all_example_names[(index * gap) % len(self.all_example_names)]

                cat_dataset = self.all_datasets[category]

                if self.is_train:
                    get_idx = random.choice(cat_dataset.seq_name2idx[sequence_name])
                elif self.args.run_viz:
                    get_idx = cat_dataset.seq_name2idx[sequence_name][
                        self.seen_frames[sequence_name]
                    ]
                    self.seen_frames[sequence_name] += 5  # NOTE(jwu): For evaluation, take very 5th frame.
                    self.seen_frames[sequence_name] = min(self.seen_frames[sequence_name],
                                                          len(cat_dataset.seq_name2idx[sequence_name])-1)
                # TODO(jwu): Currently no way to index by frame number since argument "index" is just a counter
                else:
                    get_idx = cat_dataset.seq_name2idx[sequence_name][
                        hash(sequence_name) % len(cat_dataset.seq_name2idx[sequence_name])
                    ]
                frame_data = cat_dataset.__getitem__(get_idx)
                frame_number = frame_data.frame_number
                seen_idx = None

                frame_data = cat_dataset.frame_data_type.collate([frame_data])
                seen_rgb = frame_data.image_rgb.clone().detach()

                # 112, 112, 3
                # NOTE(jwu): A hand depth map is expected, thus no masking.
                seen_xyz = dataset_utils.get_rgbd_points(
                    112, 112,
                    frame_data.camera,
                    frame_data.depth_map,
                )
                if not self.args.run_viz:
                    max_points = 20000
                    obj_point_cloud = dataset_utils._load_pointcloud(f'{self.args.mccho_path}/{category}/{sequence_name}/obj_pointclouds/pointcloud_%06d.ply' % frame_data.frame_number, max_points=max_points)
                    hand_point_cloud = dataset_utils._load_pointcloud(f'{self.args.mccho_path}/{category}/{sequence_name}/hand_pointclouds/pointcloud_%06d.ply' % frame_data.frame_number, max_points=max_points)

                    full_point_cloud = combine_point_clouds(obj_point_cloud, hand_point_cloud)
                    full_point_cloud = pad_point_cloud(full_point_cloud, max_points)

                # Normalize RGBD
                seen_xyz, mean, sd = normalize(seen_xyz)

                # Save out standardization if visualizing
                if self.args.run_viz:
                    np.save(os.path.join(self.args.job_dir, 'mean_%s_%s_%d.npy' % (category, sequence_name, frame_data.frame_number)), mean.detach().cpu().numpy())
                    np.save(os.path.join(self.args.job_dir, 'std_%s_%s_%d.npy' % (category, sequence_name, frame_data.frame_number)), sd.detach().cpu().numpy())
                break
            except Exception as e:
                print(category, sequence_name, 'sampling failed', retry, e)

        seen_rgb = seen_rgb.squeeze(0)
        if full_point_cloud is not None:
            full_rgb = full_point_cloud._features_list[0]
            # Also normalize gt
            full_points = full_point_cloud._points_list[0]
            full_points, _, _ = normalize(full_points, mean, sd)
            # Repeat for hand and object
            hand_points = hand_point_cloud._points_list[0]
            hand_points, _, _ = normalize(hand_points, mean, sd)
            # Object
            obj_points = obj_point_cloud._points_list[0]
            obj_points, _, _ = normalize(obj_points, mean, sd)

            return (
                (seen_xyz, seen_rgb),
                (full_points, full_rgb),
                frame_number,
                (category, sequence_name, seen_idx),
                (hand_points, obj_points),
            )
        elif self.args.run_viz:
            print('No gt at test time!')
            return (
                (seen_xyz, seen_rgb),
                (torch.zeros((20000, 3)), torch.zeros((20000, 3))),
                frame_number,
                (category, sequence_name, seen_idx),
                (torch.zeros((20000, 3)), torch.zeros((20000, 3))),
            )
        else:
            return (
                ([], []),
                ([], []),
                -1,
                ([], [], []),
                ([], []),
            )

    def __len__(self) -> int:
        n_objs = sum([len(cat_dataset.seq_name2idx.keys()) for cat_dataset in self.all_datasets.values()])
        if self.is_train:
            n_sequences = sum([len(cat_dataset.seq_name2idx.keys()) for cat_dataset in self.all_datasets.values()])
            # n_frames = sum([len(cat_dataset) for cat_dataset in self.all_datasets.values()])
            return n_sequences
        elif self.is_viz:
            # Test-time evaluation
            if self.args.run_viz:
                n_frames = sum([len(cat_dataset) for cat_dataset in self.all_datasets.values()])
                print('Frames:', n_frames)
                return n_frames
            # Visualization during training
            else:
                return n_objs
        else:
            return int(n_objs * self.args.eval_epoch_len_multiplier)
