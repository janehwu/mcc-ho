# This source code is adapted from:
# MCC: https://github.com/facebookresearch/MCC
import numpy as np
import cv2
import json
from tqdm import tqdm

import torch
from pytorch3d.io.obj_io import load_obj

import main_mccho
import mccho_model
import util.misc as misc
from engine_mccho import prepare_data, generate_html

# Rasterizer
import pytorch3d
from pytorch3d.structures import Meshes

# rendering components
from pytorch3d.renderer import (
    PerspectiveCameras, RasterizationSettings, MeshRasterizer, TexturesVertex
)


def run_viz(model, samples, device, args, prefix):
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
    with open(prefix + '.html', 'a') as f:
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
            score_thresholds=args.score_thresholds
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
    return seen_xyz


def main(args):

    model = mccho_model.get_mccho_model(
        occupancy_weight=1.0,
        rgb_weight=0.01,
        args=args,
    ).cuda()

    misc.load_model(args=args, model_without_ddp=model, optimizer=None, loss_scaler=None)

    rgb = cv2.imread(args.image)
    hand_obj = load_obj(args.hand)
    hand_verts = hand_obj[0]
    hand_faces = hand_obj[1].verts_idx

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

    # Rasterize hand to get visible points
    raster_settings = RasterizationSettings(
        image_size=(H, W),
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0,
    )
    rasterizer = MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    )

    verts_rgb = torch.ones(hand_verts.shape, dtype=torch.float32)
    textures = TexturesVertex(verts_features=[verts_rgb])
    hand_mesh = Meshes(
        verts=[hand_verts],
        faces=[hand_faces],
        textures=textures
    )
    fragments = rasterizer(hand_mesh)
    pix_to_face = fragments.pix_to_face
    raster_vert_weights = fragments.bary_coords
    raster_vert_indices = hand_faces[pix_to_face][..., 0, :]
    pixel_tris = hand_verts[raster_vert_indices[0].long()]
    pixel_points = torch.matmul(raster_vert_weights[0], pixel_tris)[..., 0, :]
    depth = pixel_points[..., 2]
    depth[depth < 0] = float('inf')
    print('Depth:', depth.shape)

    seen_xyz = pixel_points
    seen_xyz[..., 2] = depth
    seen_xyz = normalize(seen_xyz)

    # Prepare RGB data
    seen_rgb = (torch.tensor(rgb).float() / 255)[..., [2, 1, 0]]
    seen_rgb = torch.nn.functional.interpolate(
        seen_rgb.permute(2, 0, 1)[None],
        size=[H, W],
        mode="bilinear",
        align_corners=False,
    )[0].permute(1, 2, 0)

    # Load hand-object mask to determine bbox only
    seg = cv2.imread(args.seg, cv2.IMREAD_UNCHANGED)
    mask = torch.tensor(cv2.resize(seg, (W, H))).bool()

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
    run_viz(model, samples, "cuda", args, prefix=args.output)


if __name__ == '__main__':
    parser = main_mccho.get_args_parser()
    parser.add_argument('--image', default='demo/boardgame_v_W_qdSiPKSdQ_frame000019.jpg', type=str, help='input image file')
    parser.add_argument('--hand', default='demo/boardgame_v_W_qdSiPKSdQ_frame000019_hand.obj', type=str, help='input hand obj file')
    parser.add_argument('--seg', default='demo/boardgame_v_W_qdSiPKSdQ_frame000019_mask.png', type=str, help='input hand-object bbox')
    parser.add_argument('--cam', default='demo/camera_intrinsics_mow.json', type=str, help='input Pytorch3D PerspectiveCamera intrinsics')
    parser.add_argument('--output', default='demo/output', type=str, help='output path')
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

