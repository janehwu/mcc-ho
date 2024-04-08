# This source code is adapted from:
# MCC: https://github.com/facebookresearch/MCC
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------
import math
from typing import Iterable
import os
import matplotlib.pyplot as plt
import random
import torch
import trimesh
import numpy as np
import time
import base64
from io import BytesIO

import util.misc as misc
import util.lr_sched as lr_sched

from pytorch3d.structures import Pointclouds, Meshes
from pytorch3d.vis.plotly_vis import plot_scene
from pytorch3d.transforms import RotateAxisAngle
from pytorch3d.renderer import TexturesVertex


def write_pred_obj(out_f, points, features):
    with open(out_f, 'w') as f:
        for i in range(len(points)):
            x = points[i,0]
            y = points[i,1]
            z = points[i,2]
            r = features[i,0]
            g = features[i,1]
            b = features[i,2]
            f.write('v %f %f %f %f %f %f\n' % (x,y,z,r,g,b))


def evaluate_points(predicted_xyz, gt_xyz, dist_thres):
    if predicted_xyz.shape[0] == 0:
        return 0.0, 0.0, 0.0
    slice_size = 1000
    precision = 0.0
    for i in range(int(np.ceil(predicted_xyz.shape[0] / slice_size))):
        start = slice_size * i
        end   = slice_size * (i + 1)
        dist = ((predicted_xyz[start:end, None] - gt_xyz[None]) ** 2.0).sum(axis=-1) ** 0.5
        precision += ((dist < dist_thres).sum(axis=1) > 0).sum()
    precision /= predicted_xyz.shape[0]

    recall = 0.0
    for i in range(int(np.ceil(predicted_xyz.shape[0] / slice_size))):
        start = slice_size * i
        end   = slice_size * (i + 1)
        dist = ((predicted_xyz[:, None] - gt_xyz[None, start:end]) ** 2.0).sum(axis=-1) ** 0.5
        recall += ((dist < dist_thres).sum(axis=0) > 0).sum()
    recall /= gt_xyz.shape[0]
    return precision, recall, get_f1(precision, recall)

def aug_xyz(seen_xyz, unseen_xyz, args, is_train):
    degree_x = 0
    degree_y = 0
    degree_z = 0
    if is_train:
        r_delta = args.random_scale_delta
        scale = torch.tensor([
            random.uniform(1.0 - r_delta, 1.0 + r_delta),
            random.uniform(1.0 - r_delta, 1.0 + r_delta),
            random.uniform(1.0 - r_delta, 1.0 + r_delta),
        ], device=seen_xyz.device)

        degree_x = random.randrange(-args.random_rotate_degree, args.random_rotate_degree + 1)
        degree_y = random.randrange(-args.random_rotate_degree, args.random_rotate_degree + 1)
        degree_z = random.randrange(-args.random_rotate_degree, args.random_rotate_degree + 1)

        r_shift = args.random_shift
        shift = torch.tensor([[[
            random.uniform(-r_shift, r_shift),
            random.uniform(-r_shift, r_shift),
            random.uniform(-r_shift, r_shift),
        ]]], device=seen_xyz.device)
        seen_xyz = seen_xyz * scale + shift
        unseen_xyz = unseen_xyz * scale + shift

    B, H, W, _ = seen_xyz.shape
    return [
        rotate(seen_xyz.reshape((B, -1, 3)), degree_x, degree_y, degree_z).reshape((B, H, W, 3)),
        rotate(unseen_xyz, degree_x, degree_y, degree_z),
    ]


def rotate(sample, degree_x, degree_y, degree_z):
    for degree, axis in [(degree_x, "X"), (degree_y, "Y"), (degree_z, "Z")]:
        if degree != 0:
            sample = RotateAxisAngle(degree, axis=axis).to(sample.device).transform_points(sample)
    return sample


def get_grid(B, device, mccho_world_size, granularity):
    N = int(np.ceil(2 * mccho_world_size / granularity))
    grid_unseen_xyz = torch.zeros((N, N, N, 3), device=device)
    for i in range(N):
        grid_unseen_xyz[i, :, :, 0] = i
    for j in range(N):
        grid_unseen_xyz[:, j, :, 1] = j
    for k in range(N):
        grid_unseen_xyz[:, :, k, 2] = k
    grid_unseen_xyz -= (N / 2.0)
    grid_unseen_xyz /= (N / 2.0) / mccho_world_size
    grid_unseen_xyz = grid_unseen_xyz.reshape((1, -1, 3)).repeat(B, 1, 1)
    return grid_unseen_xyz


def get_refined_grid(B, device, mccho_world_size, granularity):
    grid_dims = mccho_world_size[1] - mccho_world_size[0]
    print('Grid dims:', grid_dims)
    N = np.ceil(grid_dims / granularity).astype(np.int64)
    grid_unseen_xyz = torch.zeros((N[0], N[1], N[2], 3), device=device)
    print('GRID:', grid_unseen_xyz.shape)
    for i in range(N[0]):
        grid_unseen_xyz[i, :, :, 0] = mccho_world_size[0][0] + granularity*i
    for j in range(N[1]):
        grid_unseen_xyz[:, j, :, 1] = mccho_world_size[0][1] + granularity*j
    for k in range(N[2]):
        grid_unseen_xyz[:, :, k, 2] = mccho_world_size[0][2] + granularity*k
    #grid_unseen_xyz -= (N / 2.0)
    #grid_unseen_xyz /= (N / 2.0) / mccho_world_size
    grid_unseen_xyz = grid_unseen_xyz.reshape((1, -1, 3)).repeat(B, 1, 1)
    return grid_unseen_xyz


def run_viz(model, data_loader, device, args, epoch):
    epoch_start_time = time.time()
    model.eval()
    os.system(f'mkdir {args.job_dir}/viz')

    print('Visualization data_loader length:', len(data_loader))
    dataset = data_loader.dataset
    for sample_idx, samples in enumerate(data_loader):
        frame_number = samples[2][0]
        if frame_number == -1:
            print('SKIPPING INVALID FRAME')
            continue
        category = samples[3][0][0]
        sequence_name = samples[3][0][1]
        print('Frame info:', category, sequence_name, frame_number, sample_idx)

        if sample_idx >= args.max_n_viz_obj:
            break
        start = time.time()
        seen_xyz, valid_seen_xyz, unseen_xyz, unseen_rgb, labels, seen_images, unseen_seg = prepare_data(samples, device, is_train=False, args=args, is_viz=True)
        print('Prepared data:', time.time()-start)
        pred_occupy = []
        pred_colors = []
        pred_segs = []
        (model.module if hasattr(model, "module") else model).clear_cache()

        # don't forward all at once to avoid oom
        max_n_queries_fwd = 10000
        start = time.time()
        total_n_passes = int(np.ceil(unseen_xyz.shape[1] / max_n_queries_fwd))
        for p_idx in range(total_n_passes):
            p_start = p_idx     * max_n_queries_fwd
            p_end = (p_idx + 1) * max_n_queries_fwd
            cur_unseen_xyz = unseen_xyz[:, p_start:p_end]
            cur_unseen_rgb = unseen_rgb[:, p_start:p_end].zero_()
            cur_labels = labels[:, p_start:p_end].zero_()
            cur_unseen_seg = None
            if args.segmentation_label:
                cur_unseen_seg = unseen_seg[:, p_start:p_end].zero_()
            with torch.no_grad():
                _, pred, = model(
                    seen_images=seen_images,
                    seen_xyz=seen_xyz,
                    unseen_xyz=cur_unseen_xyz,
                    unseen_rgb=cur_unseen_rgb,
                    unseen_occupy=cur_labels,
                    cache_enc=args.run_viz,
                    valid_seen_xyz=valid_seen_xyz,
                    unseen_seg=cur_unseen_seg,
                )

            cur_occupy_out = pred[..., 0]

            if args.regress_color:
                color_values = 3
                cur_color_out = pred[..., 1:color_values+1].reshape((-1, 3))
            else:
                color_values = 256 * 3
                cur_color_out = pred[..., 1:color_values+1].reshape((-1, 3, 256)).max(dim=2)[1] / 255.0

            pred_occupy.append(cur_occupy_out)
            pred_colors.append(cur_color_out)

            cur_seg_out = None
            if args.segmentation_label:
                assert pred.shape[-1] == 1 + color_values + 3
                cur_seg_out = pred[..., -3:].reshape((-1, 3)).max(dim=1)[1]
                pred_segs.append(cur_seg_out)
        print('Model inference done:', time.time() - start)
        rank = misc.get_rank()
        prefix = f'{args.job_dir}/viz/{category}_{sequence_name}/' + dataset.dataset_split + f'_ep{epoch}_rank{rank}_{category}_{sequence_name}_{frame_number}'
        os.makedirs(f'{args.job_dir}/viz/{category}_{sequence_name}', exist_ok=True)
        img = (seen_images[0].permute(1, 2, 0) * 255).cpu().numpy().copy().astype(np.uint8)

        gt_xyz = samples[1][0].to(device).reshape(-1, 3)
        gt_rgb = samples[1][1].to(device).reshape(-1, 3)
        # Visualization during training
        if args.run_viz:
            generate_objs(
                torch.cat(pred_occupy, dim=1),
                torch.cat(pred_colors, dim=0),
                unseen_xyz,
                prefix,
                pred_seg=torch.cat(pred_segs, dim=0) if args.segmentation_label else None,
                score_thresholds=[0.1]
            )
        with open(prefix + '.html', 'a') as f:
            generate_html(
                img,
                seen_xyz, seen_images,
                torch.cat(pred_occupy, dim=1),
                torch.cat(pred_colors, dim=0),
                unseen_xyz,
                f,
                gt_xyz=gt_xyz,
                gt_rgb=gt_rgb,
                pred_seg=torch.cat(pred_segs, dim=0) if args.segmentation_label else None,
                score_thresholds=[0.1, 0.3, 0.5]
            )
    print("Visualization epoch time:", time.time() - epoch_start_time)


def get_f1(precision, recall):
    if (precision + recall) == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def generate_objs(pred_occ, pred_rgb, unseen_xyz, prefix, pred_seg=None,
        score_thresholds=[0.1, 0.3, 0.5, 0.7, 0.9]):
    clouds = {"MCC Output": {}}
    pred_occ = torch.nn.Sigmoid()(pred_occ).cpu()
    for t in score_thresholds:
        pos = pred_occ > t

        points = unseen_xyz[pos].reshape((-1, 3))
        features = pred_rgb[None][pos].reshape((-1, 3))
        good_points = points[:, 0] != -100

        if good_points.sum() == 0:
            print('No points')
            return
        points = points[good_points].cpu()
        features = features[good_points].cpu()

        # Write obj (just pts)
        out_f = prefix + ('_%.1f.obj' % t)
        print('Writing obj to:', out_f)
        write_pred_obj(out_f, points, features)

        # Write hand and obj if segmented
        if pred_seg is not None:
            seg = pred_seg[None][pos].reshape((-1,))
            hand_idx = seg == 1
            obj_idx = seg != 1
            points = unseen_xyz[pos].reshape((-1, 3))
            features = pred_rgb[None][pos].reshape((-1, 3))

            # Hand
            hand_points = points[hand_idx]
            hand_features = features[hand_idx]
            good_points = hand_points[:, 0] != -100
            hand_points = hand_points[good_points].cpu()
            hand_features = hand_features[good_points].cpu()
            if len(hand_points) > 0:
                out_f = prefix + ('_%.1f_hand.obj' % t)
                write_pred_obj(out_f, hand_points, hand_features)

            # Obj
            obj_points = points[obj_idx]
            obj_features = features[obj_idx]
            good_points = obj_points[:, 0] != -100
            obj_points = obj_points[good_points].cpu()
            obj_features = obj_features[good_points].cpu()
            if len(obj_points) > 0:
                out_f = prefix + ('_%.1f_obj.obj' % t)
                write_pred_obj(out_f, obj_points, obj_features)
    return


def generate_html(img, seen_xyz, seen_rgb, pred_occ, pred_rgb, unseen_xyz, f,
        gt_xyz=None, gt_rgb=None, mesh_xyz=None, pred_seg=None, hand_xyz=None,
        hand_faces=None, hand_rgb=None, score_thresholds=[0.1, 0.3, 0.5, 0.7, 0.9],
        pointcloud_marker_size=2, device='cuda'
    ):
    if img is not None:
        fig = plt.figure()
        plt.imshow(img)
        tmpfile = BytesIO()
        fig.savefig(tmpfile, format='jpg')
        encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')

        html = '<img src=\'data:image/png;base64,{}\'>'.format(encoded)
        f.write(html)
        plt.close()

    clouds = {"MCC Output": {}}
    # Seen
    if seen_xyz is not None:
        seen_xyz = seen_xyz.reshape((-1, 3)).cpu()
        seen_rgb = torch.nn.functional.interpolate(seen_rgb, (112, 112)).permute(0, 2, 3, 1).reshape((-1, 3)).cpu()
        good_seen = seen_xyz[:, 0] != -100

        seen_pc = Pointclouds(
            points=seen_xyz[good_seen][None],
            features=seen_rgb[good_seen][None],
        )
        clouds["MCC Output"]["seen"] = seen_pc

    # GT points
    if gt_xyz is not None:
        subset_gt = random.sample(range(gt_xyz.shape[0]), 10000)
        gt_pc = Pointclouds(
            points=gt_xyz[subset_gt][None],
            features=gt_rgb[subset_gt][None],
        )
        clouds["MCC Output"]["GT points"] = gt_pc

    # GT meshes
    if mesh_xyz is not None:
        subset_mesh = random.sample(range(mesh_xyz.shape[0]), 10000)
        mesh_pc = Pointclouds(
            points=mesh_xyz[subset_mesh][None],
        )
        clouds["MCC Output"]["GT mesh"] = mesh_pc

    # Hand meshes
    if hand_xyz is not None:
        textures = TexturesVertex(verts_features=[c.to(device) for c in hand_rgb])

        mesh_hand = Meshes(
            verts=[v.to(device) for v in hand_xyz],
            faces=[f.to(device) for f in hand_faces],
            textures=textures
        )
        clouds["MCC Output"]["Hand"] = mesh_hand

    pred_occ = torch.nn.Sigmoid()(pred_occ).cpu()
    for t in score_thresholds:
        pos = pred_occ > t

        points = unseen_xyz[pos].reshape((-1, 3))
        features = pred_rgb[None][pos].reshape((-1, 3))
        good_points = points[:, 0] != -100

        if good_points.sum() == 0:
            continue

        pc = Pointclouds(
            points=points[good_points][None].cpu(),
            features=features[good_points][None].cpu(),
        )

        clouds["MCC Output"][f"pred_{t}"] = pc

        if pred_seg is not None and len(pred_seg) > 0:
            # Add hand and object segmentations
            seg = pred_seg[None][pos].reshape((-1,))
            assert len(seg) == len(features)
            hand_points = seg == 1
            obj_points = seg == 2
            if hand_points.sum() > 0:
                hand_pc = Pointclouds(
                    points=points[hand_points][None].cpu(),
                    features=features[hand_points][None].cpu(),
                )
                clouds["MCC Output"][f"pred_hand_{t}"] = hand_pc
            if obj_points.sum() > 0:
                obj_pc = Pointclouds(
                    points=points[obj_points][None].cpu(),
                    features=features[obj_points][None].cpu(),
                )
                clouds["MCC Output"][f"pred_obj_{t}"] = obj_pc

    plt.figure()
    try:
        fig = plot_scene(clouds, pointcloud_marker_size=pointcloud_marker_size, pointcloud_max_points=20000 * 2)
        fig.update_layout(height=1000, width=1000)
        f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
    except Exception as e:
        print('writing failed', e)
    try:
        plt.close()
    except:
        pass


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    args=None):
    epoch_start_time = time.time()
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    print('Training data_loader length:', len(data_loader))
    for data_iter_step, samples in enumerate(data_loader):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        seen_xyz, valid_seen_xyz, unseen_xyz, unseen_rgb, labels, seen_images, unseen_seg = prepare_data(samples, device, is_train=True, args=args)

        with torch.cuda.amp.autocast():
            loss, _ = model(
                seen_images=seen_images,
                seen_xyz=seen_xyz,
                unseen_xyz=unseen_xyz,
                unseen_rgb=unseen_rgb,
                unseen_occupy=labels,
                valid_seen_xyz=valid_seen_xyz,
                unseen_seg=unseen_seg,
            )

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Warning: Loss is {}".format(loss_value))
            loss *= 0.0
            loss_value = 100.0

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    clip_grad=args.clip_grad,
                    update_grad=(data_iter_step + 1) % accum_iter == 0,
                    verbose=(data_iter_step % 100) == 0)

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        if data_iter_step == 30:
            os.system('nvidia-smi')
            os.system('free -g')
        if args.debug and data_iter_step == 5:
            break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    print("Training epoch time:", time.time() - epoch_start_time)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def eval_one_epoch(
        model: torch.nn.Module,
        data_loader: Iterable,
        device: torch.device,
        args=None
    ):
    epoch_start_time = time.time()
    model.train(False)

    metric_logger = misc.MetricLogger(delimiter="  ")

    print('Eval len(data_loader):', len(data_loader))

    for data_iter_step, samples in enumerate(data_loader):
        frame_number = samples[2][0]
        category = samples[3][0]
        sequence_name = samples[3][0]
        print('Frame info:', category, sequence_name, frame_number, data_iter_step)
        if data_iter_step >= args.max_n_eval:
            break

        seen_xyz, valid_seen_xyz, unseen_xyz, unseen_rgb, labels, seen_images, unseen_seg = prepare_data(samples, device, is_train=False, args=args)

        # don't forward all at once to avoid oom
        max_n_queries_fwd = 5000
        all_loss, all_preds = [], []
        for p_idx in range(int(np.ceil(unseen_xyz.shape[1] / max_n_queries_fwd))):
            p_start = p_idx     * max_n_queries_fwd
            p_end = (p_idx + 1) * max_n_queries_fwd
            cur_unseen_xyz = unseen_xyz[:, p_start:p_end]
            cur_unseen_rgb = unseen_rgb[:, p_start:p_end]
            cur_labels = labels[:, p_start:p_end]
            cur_unseen_seg = None
            if unseen_seg is not None:
                cur_unseen_seg = unseen_seg[:, p_start:p_end]

            with torch.no_grad():
                loss, pred = model(
                    seen_images=seen_images,
                    seen_xyz=seen_xyz,
                    unseen_xyz=cur_unseen_xyz,
                    unseen_rgb=cur_unseen_rgb,
                    unseen_occupy=cur_labels,
                    valid_seen_xyz=valid_seen_xyz,
                    unseen_seg=cur_unseen_seg,
                )
            all_loss.append(loss)
            all_preds.append(pred)

        loss = sum(all_loss) / len(all_loss)
        pred = torch.cat(all_preds, dim=1)

        B = pred.shape[0]

        gt_xyz = samples[1][0].to(device).reshape((B, -1, 3))

        s_thres = args.eval_score_threshold
        d_thres = args.eval_dist_threshold

        for b_idx in range(B):
            geometry_metrics = {}
            predicted_idx = torch.nn.Sigmoid()(pred[b_idx, :, 0]) > s_thres
            predicted_xyz = unseen_xyz[b_idx, predicted_idx]

            precision, recall, f1 = evaluate_points(predicted_xyz, gt_xyz[b_idx], d_thres)
            geometry_metrics[f'd{d_thres}_s{s_thres}_point_pr'] = precision
            geometry_metrics[f'd{d_thres}_s{s_thres}_point_rc'] = recall
            geometry_metrics[f'd{d_thres}_s{s_thres}_point_f1'] = f1

            metric_logger.update(**geometry_metrics)

        loss_value = loss.item()

        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)

        if args.debug and data_iter_step == 5:
            break

    metric_logger.synchronize_between_processes()
    print("Validation averaged stats:", metric_logger)
    print("Val epoch time:", time.time() - epoch_start_time)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def sample_uniform_semisphere(B, N, semisphere_size, device):
    for _ in range(100):
        points = torch.empty(B * N * 3, 3, device=device).uniform_(-semisphere_size, semisphere_size)
        points[..., 2] = points[..., 2].abs()
        dist = (points ** 2.0).sum(axis=-1) ** 0.5
        if (dist < semisphere_size).sum() >= B * N:
            return points[dist < semisphere_size][:B * N].reshape((B, N, 3))
        else:
            print('resampling sphere')


def get_min_dist(a, b, slice_size=1000):
    all_min, all_idx = [], []
    for i in range(int(np.ceil(a.shape[1] / slice_size))):
        start = slice_size * i
        end   = slice_size * (i + 1)
        # B, n_queries, n_gt
        dist = ((a[:, start:end] - b) ** 2.0).sum(axis=-1) ** 0.5
        # B, n_queries
        cur_min, cur_idx = dist.min(axis=2)
        all_min.append(cur_min)
        all_idx.append(cur_idx)
    return torch.cat(all_min, dim=1), torch.cat(all_idx, dim=1)


def construct_uniform_grid(gt_xyz, gt_rgb, mccho_world_size, n_queries, dist_threshold, is_train, granularity,
    hand_xyz=None, obj_xyz=None):
    B = gt_xyz.shape[0]
    device = gt_xyz.device
    if is_train:
        unseen_xyz = torch.empty((B, n_queries, 3), device=device).uniform_(-mccho_world_size, mccho_world_size)
    elif type(mccho_world_size) != float:
        unseen_xyz = get_refined_grid(B, device, mccho_world_size, granularity)
    else:
        unseen_xyz = get_grid(B, device, mccho_world_size, granularity)
    dist, idx_to_gt = get_min_dist(unseen_xyz[:, :, None], gt_xyz[:, None])
    labels = dist < dist_threshold

    # Hand and obj
    hand_labels = None
    obj_labels = None
    if hand_xyz is not None:
        hand_dist, _ = get_min_dist(unseen_xyz[:, :, None], hand_xyz[:, None])
        hand_labels = hand_dist < dist_threshold
        hand_labels = hand_labels.float()
    if obj_xyz is not None:
        obj_dist, _ = get_min_dist(unseen_xyz[:, :, None], obj_xyz[:, None])
        obj_labels = obj_dist < dist_threshold
        labels = torch.logical_or(hand_labels, obj_labels)
        obj_labels.float()

    unseen_rgb = torch.zeros_like(unseen_xyz)
    unseen_rgb[labels] = torch.gather(gt_rgb, 1, idx_to_gt.unsqueeze(-1).repeat(1, 1, 3))[labels]
    return unseen_xyz, unseen_rgb, labels.float(), hand_labels, obj_labels


def prepare_data(samples, device, is_train, args, is_viz=False):
    # Seen
    seen_xyz, seen_rgb = samples[0][0].to(device), samples[0][1].to(device)
    valid_seen_xyz = torch.isfinite(seen_xyz.sum(axis=-1))
    seen_xyz[~valid_seen_xyz] = -100
    B = seen_xyz.shape[0]
    # Gt
    gt_xyz, gt_rgb = samples[1][0].to(device).reshape(B, -1, 3), samples[1][1].to(device).reshape(B, -1, 3)

    sampling_func = construct_uniform_grid
    if args.segmentation_label:
        hand_xyz, obj_xyz = samples[4][0].to(device).reshape(B, -1, 3), samples[4][1].to(device).reshape(B, -1, 3)

    # Set grid parameters
    if not args.refine_grid:
        granularity = args.viz_granularity if is_viz else args.eval_granularity
        mccho_world_size = args.mccho_world_size
    else:
        # Get example
        frame_number = samples[2][0]
        category, sequence_name, seen_idx = samples[3][0]
        print('SAMPLE:', frame_number, category, sequence_name)
        # Load previous prediction
        prefix = f'{args.job_dir}/viz/{category}_{sequence_name}/test_ep0_rank0_{category}_{sequence_name}_{frame_number}'
        obj_fname = prefix.replace('_refine', '') + '_0.1.obj'
        print(obj_fname)
        if not os.path.exists(obj_fname):
            print('FALLING BACK')
            granularity = args.viz_granularity if is_viz else args.eval_granularity
            mccho_world_size = args.mccho_world_size
        else:
            obj_verts = np.asarray(trimesh.load(obj_fname).vertices)
            grid_dims = np.max(obj_verts, axis=0) - np.min(obj_verts, axis=0)
            granularity = max(grid_dims) / 50.
            mccho_world_size = [np.min(obj_verts, axis=0), np.max(obj_verts, axis=0)]
            print('World size:', granularity, mccho_world_size)
    unseen_xyz, unseen_rgb, labels, hand_labels, obj_labels = sampling_func(
        gt_xyz, gt_rgb,
        mccho_world_size,
        args.n_queries,
        args.train_dist_threshold,
        is_train,
        granularity,
        hand_xyz=hand_xyz if args.segmentation_label else None,
        obj_xyz=obj_xyz if args.segmentation_label else None,
    )

    unseen_seg = None
    if args.segmentation_label:
        # Combine labels for gt segmentation
        background_labels = 1 - torch.clip(hand_labels + obj_labels, 0, 1)
        unseen_seg = torch.cat([background_labels[..., None],
                                hand_labels[..., None],
                                obj_labels[..., None]], axis=-1)
        unseen_seg = unseen_seg.max(dim=-1)[1]
        foreground = (unseen_seg > 0).float()
        # Check that segmentation and occupancy are consistent
        assert torch.equal(foreground, labels)

    if is_train:
        seen_xyz, unseen_xyz = aug_xyz(seen_xyz, unseen_xyz, args, is_train=is_train)

        # Random Flip
        if random.random() < 0.5:
            seen_xyz[..., 0] *= -1
            unseen_xyz[..., 0] *= -1
            seen_xyz = torch.flip(seen_xyz, [2])
            valid_seen_xyz = torch.flip(valid_seen_xyz, [2])
            seen_rgb = torch.flip(seen_rgb, [3])

    return seen_xyz, valid_seen_xyz, unseen_xyz, unseen_rgb, labels, seen_rgb, unseen_seg
