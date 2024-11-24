# This source code is adapted from:
# MCC: https://github.com/facebookresearch/MCC
import os
import numpy as np
import cv2
import json
from tqdm import tqdm

import torch
import trimesh
from pytorch3d.io.obj_io import load_obj

import main_mccho
import mccho_model
import util.hamer_utils as hamer
import util.misc as misc
from engine_mccho import prepare_data, generate_html, generate_objs

# Rasterizer
import pytorch3d
from pytorch3d.structures import Meshes

# rendering components
from pytorch3d.renderer import (
    BlendParams, PerspectiveCameras, RasterizationSettings, MeshRasterizer,
    MeshRenderer, SoftSilhouetteShader, TexturesVertex
)


def get_silhouette_renderer(cameras, imh, imw):
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)

    raster_settings = RasterizationSettings(
        image_size=(imh, imw), 
        blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma, 
        faces_per_pixel=150,
        bin_size=0,
    )
    
    # Create a silhouette mesh renderer by composing a rasterizer and a shader. 
    silhouette_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=SoftSilhouetteShader(blend_params=blend_params)
    )
    return silhouette_renderer


def get_rasterizer(cameras, imh, imw):
    raster_settings = RasterizationSettings(
        image_size=(imh, imw),
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0,
    )
    rasterizer = MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    )
    return rasterizer


def run_viz(model, samples, device, args, seen_mean, seen_sd, prefix):
    model.eval()

    seen_xyz, valid_seen_xyz, unseen_xyz, unseen_rgb, labels, seen_images, unseen_seg = prepare_data(
        samples, device, is_train=False, args=args, is_viz=True
    )
    pred_occupy = []
    pred_colors = []
    pred_segs = []
    max_n_unseen_fwd = 2000

    model.cached_enc_feat = None
    num_passes = int(np.ceil(unseen_xyz.shape[1] / max_n_unseen_fwd))
    for p_idx in tqdm(range(num_passes)):
        p_start = p_idx     * max_n_unseen_fwd
        p_end = (p_idx + 1) * max_n_unseen_fwd
        cur_unseen_xyz = unseen_xyz[:, p_start:p_end]
        cur_unseen_rgb = unseen_rgb[:, p_start:p_end].zero_()
        cur_labels = labels[:, p_start:p_end].zero_()
        cur_unseen_seg = unseen_seg[:, p_start:p_end].zero_()

        with torch.no_grad():
            _, pred = model(
                seen_images=seen_images,
                seen_xyz=seen_xyz,
                unseen_xyz=cur_unseen_xyz,
                unseen_rgb=cur_unseen_rgb,
                unseen_occupy=cur_labels,
                cache_enc=True,
                valid_seen_xyz=valid_seen_xyz,
                unseen_seg=cur_unseen_seg,
            )
        pred_occupy.append(pred[..., 0].cpu())
        if args.regress_color:
            color_values = 3
            pred_colors.append(pred[..., 1:color_values+1].reshape((-1, 3)))
        else:
            color_values = 256 * 3
            pred_colors.append(
                (
                    torch.nn.Softmax(dim=2)(
                        pred[..., 1:color_values+1].reshape((-1, 3, 256)) / args.temperature
                    ) * torch.linspace(0, 1, 256, device=pred.device)
                ).sum(axis=2)
            )
        assert pred.shape[-1] == 1 + color_values + 3
        cur_seg_out = pred[..., -3:].reshape((-1, 3)).max(dim=1)[1]
        pred_segs.append(cur_seg_out)

    img = (seen_images[0].permute(1, 2, 0) * 255).cpu().numpy().copy().astype(np.uint8)
    with open(prefix + '.html', 'w') as f:
        generate_html(
            img,
            seen_xyz, seen_images,
            torch.cat(pred_occupy, dim=1),
            torch.cat(pred_colors, dim=0),
            unseen_xyz,
            f,
            gt_xyz=None,
            gt_rgb=None,
            mesh_xyz=None,
            pred_seg=torch.cat(pred_segs, dim=0),
            score_thresholds=args.score_thresholds,
        )
        generate_objs(
            torch.cat(pred_occupy, dim=1),
            torch.cat(pred_colors, dim=0),
            unseen_xyz,
            prefix,
            pred_seg=torch.cat(pred_segs, dim=0) if args.segmentation_label else None,
            score_thresholds=[0.1],
            seen_mean=seen_mean,
            seen_sd=seen_sd,
        )


def pad_image(im, value):
    if im.shape[0] > im.shape[1]:
        diff = im.shape[0] - im.shape[1]
        return torch.cat([im, (torch.zeros((im.shape[0], diff, im.shape[2])) + value)], dim=1)
    else:
        diff = im.shape[1] - im.shape[0]
        return torch.cat([im, (torch.zeros((diff, im.shape[1], im.shape[2])) + value)], dim=0)


def normalize(seen_xyz, sd_scale=3):
    mean = seen_xyz[torch.isfinite(seen_xyz.sum(dim=-1))].mean(axis=0)
    sd = seen_xyz[torch.isfinite(seen_xyz.sum(dim=-1))].std(dim=0).mean() * sd_scale
    seen_xyz = (seen_xyz - mean) / sd
    return seen_xyz, mean, sd


def main(args, device='cuda'):
    # Read image
    rgb = cv2.imread(args.image)

    # Run HaMeR to obtain 3D hands
    hamer_out = hamer.run_demo(args.image, os.path.join(args.out_folder, 'hamer'))
    # Select the hand
    batch_size = hamer_out['batch']['img'].shape[0]
    hand_idx = 0
    for n in range(batch_size):
        # Hands are ranked by confidence, so we take the first match.
        if args.is_right_hand == hamer_out['batch']['right'][n]:
            hand_idx = n
            break
    hand_path = os.path.join(args.out_folder, 'hamer', args.image.split('.')[0].split('/')[-1] + '_%d.obj' % n)
    print('Selected hand:', hand_path)
    assert os.path.exists(hand_path)
    hand_obj = load_obj(hand_path)
    hand_verts = hand_obj[0]
    # pyrender to pytorch3d conversion
    hand_verts[:, 0] *= -1
    hand_verts[:, 2] *= -1
    hand_faces = hand_obj[1].verts_idx
    # Save out for debugging.
    mesh = trimesh.Trimesh(hand_verts.detach().cpu().numpy(), hand_faces.detach().cpu().numpy())
    mesh.export(os.path.join(args.out_folder, 'input_hand.obj'))

    # Load MCC-HO model.
    model = mccho_model.get_mccho_model(
        occupancy_weight=1.0,
        rgb_weight=0.01,
        args=args,
    ).cuda()

    misc.load_model(args=args, model_without_ddp=model, optimizer=None, loss_scaler=None)

    # Initialize camera
    H, W = rgb.shape[:2]
    cam_f = open(args.cam)
    intrinsics = json.load(cam_f)
    fx = intrinsics['fx']
    fy = intrinsics['fy']
    px = intrinsics['px']
    py = intrinsics['py']
    print('Camera intrinsics:', fx, fy, px, py)
    cameras = PerspectiveCameras(
        focal_length=((fx, fy),),
        principal_point=((px, py),),
        image_size=((H, W),),
        device='cpu'
    )

    # Initialize hand mesh as Pytorch3D object
    verts_rgb = torch.ones(hand_verts.shape, dtype=torch.float32)
    textures = TexturesVertex(verts_features=[verts_rgb])
    hand_mesh = Meshes(
        verts=[hand_verts],
        faces=[hand_faces],
        textures=textures
    )

    # Render hand mask to be combined with obj mask later
    silhouette_renderer = get_silhouette_renderer(cameras, H, W)
    hand_mask = silhouette_renderer(hand_mesh)[0].detach().cpu().numpy()
    hand_mask = (hand_mask[..., 3] > 0).astype(np.uint8)
    cv2.imwrite(os.path.join(args.out_folder, 'hand_mask.png'), hand_mask*255)

    # Rasterize hand to get visible points
    rasterizer = get_rasterizer(cameras, H, W)
    fragments = rasterizer(hand_mesh)
    pix_to_face = fragments.pix_to_face
    raster_vert_weights = fragments.bary_coords
    raster_vert_indices = hand_faces[pix_to_face][..., 0, :]
    pixel_tris = hand_verts[raster_vert_indices[0].long()]
    pixel_points = torch.matmul(raster_vert_weights[0], pixel_tris)[..., 0, :]
    depth = pixel_points[..., 2]
    depth[depth < 0] = float('inf')

    seen_xyz = pixel_points
    seen_xyz[..., 2] = depth
    seen_xyz, seen_mean, seen_sd = normalize(seen_xyz)

    # Prepare RGB data
    seen_rgb = (torch.tensor(rgb).float() / 255)[..., [2, 1, 0]]
    seen_rgb = torch.nn.functional.interpolate(
        seen_rgb.permute(2, 0, 1)[None],
        size=[H, W],
        mode="bilinear",
        align_corners=False,
    )[0].permute(1, 2, 0)

    # Load hand-object mask to determine bbox only
    obj_mask = (cv2.imread(args.obj_seg, cv2.IMREAD_UNCHANGED) > 0).astype(np.uint8)
    mask = ((obj_mask + hand_mask) > 0).astype(np.uint8)
    cv2.imwrite(os.path.join(args.out_folder, 'combined_mask.png'), mask*255)
    mask = torch.tensor(cv2.resize(mask, (W, H))).bool()

    bottom, right = mask.nonzero().max(dim=0)[0]
    top, left = mask.nonzero().min(dim=0)[0]

    bottom = bottom + 50
    right = right + 50
    top = max(top - 50, 0)
    left = max(left - 50, 0)

    seen_xyz = seen_xyz[top:bottom+1, left:right+1]
    seen_rgb = seen_rgb[top:bottom+1, left:right+1]

    seen_xyz = pad_image(seen_xyz, float('inf'))
    seen_rgb = pad_image(seen_rgb, 0)

    seen_rgb = torch.nn.functional.interpolate(
        seen_rgb.permute(2, 0, 1)[None],
        size=[800, 800],
        mode="bilinear",
        align_corners=False,
    )

    seen_xyz = torch.nn.functional.interpolate(
        seen_xyz.permute(2, 0, 1)[None],
        size=[112, 112],
        mode="bilinear",
        align_corners=False,
    ).permute(0, 2, 3, 1)

    samples = [
        [seen_xyz, seen_rgb],
        [torch.zeros((20000, 3)), torch.zeros((20000, 3))],
        0,
        [[], [], []],
        [torch.zeros((20000, 3)), torch.zeros((20000, 3))],

    ]
    prefix = os.path.join(args.out_folder, 'output')
    run_viz(model, samples, device, args, seen_mean, seen_sd, prefix)


if __name__ == '__main__':
    parser = main_mccho.get_args_parser()
    parser.add_argument('--image', default='demo/drink_v_1F96GArORtg_frame000084.jpg', type=str, help='input image file')
    parser.add_argument('--obj_seg', default='demo/drink_v_1F96GArORtg_frame000084_mask.png', type=str, help='input object segmentation mask')
    parser.add_argument('--cam', default='demo/camera_intrinsics_hamer.json', type=str, help='input Pytorch3D PerspectiveCamera intrinsics corresponding to scaled_focal_length=1000')
    parser.add_argument('--is_right_hand', default=True, type=bool, help='whether the right hand is manipulating the object')
    parser.add_argument('--out_folder', default='out_demo', type=str, help='output folder')
    parser.add_argument('--granularity', default=0.1, type=float, help='output granularity')
    parser.add_argument('--score_thresholds', default=[0.1, 0.2, 0.3, 0.4, 0.5], type=float, nargs='+', help='score thresholds')
    parser.add_argument('--temperature', default=0.1, type=float, help='temperature for color prediction.')
    parser.add_argument('--checkpoint', default='mccho_best_checkpoint.pth', type=str, help='model checkpoint')

    parser.set_defaults(eval=True)

    args = parser.parse_args()
    args.resume = args.checkpoint
    args.viz_granularity = args.granularity
    args.segmentation_label = True
    main(args)

