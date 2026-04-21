'''
Inference code for SgMg, on refer_youtube_vos
Modified from DETR (https://github.com/facebookresearch/detr)
refer_davis17 does not support visualize
'''
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch


import misc as utils
from models import build_model
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image, ImageDraw, ImageFont
import math
import torch.nn.functional as F
import json

from tqdm import tqdm
import shutil

import multiprocessing as mp
import threading
import warnings
warnings.filterwarnings("ignore")

from ruamel.yaml import YAML
from easydict import EasyDict
from torch.cuda.amp import autocast
from misc import nested_tensor_from_videos_list

# colormap
color_list = utils.colormap()
color_list = color_list.astype('uint8').tolist()

# build transform
transform = T.Compose([
    T.Resize(360),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

GREEN_MASK_COLOR = np.array([0, 255, 0], dtype=np.float32)
MASK_ALPHA = 0.5


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def create_video_writer(path, frame_size, fps):
    ensure_dir(os.path.dirname(path))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video writer for {path}")
    return writer


def write_overlay_video(video_path, img_folder, video_name, frames, masks, fps):
    video_writer = None
    try:
        for frame_name, mask in zip(frames, masks):
            img_path = os.path.join(img_folder, video_name, frame_name + ".jpg")
            source_img = Image.open(img_path).convert('RGB')
            overlay_frame = vis_add_mask(source_img, mask, GREEN_MASK_COLOR, alpha=MASK_ALPHA)
            overlay_bgr = cv2.cvtColor(np.asarray(overlay_frame), cv2.COLOR_RGB2BGR)
            if video_writer is None:
                frame_h, frame_w = overlay_bgr.shape[:2]
                video_writer = create_video_writer(video_path, (frame_w, frame_h), fps)
            video_writer.write(overlay_bgr)
    finally:
        if video_writer is not None:
            video_writer.release()


def main(args):
    print("Inference only supports for batch size = 1")
    print(args)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    split = "valid"
    # save path
    save_dir = os.path.join(args.output_dir, args.dataset_name, args.version)
    output_dir = os.path.join(save_dir, "Annotations")
    os.makedirs(output_dir, exist_ok=True)
    shutil.copyfile(src=args.config_path, dst=os.path.join(save_dir, 'config.yaml'))

    save_visualize_path_prefix = os.path.join(save_dir, split + '_images')
    if args.visualize:
        if not os.path.exists(save_visualize_path_prefix):
            os.makedirs(save_visualize_path_prefix)

    # load data
    root = Path(args.dataset_path) # data/refer_davis
    img_folder = os.path.join(root, split, "JPEGImages")
    meta_file = os.path.join(root, "meta_expressions", split, "meta_expressions.json")
    with open(meta_file, "r") as f:
        data = json.load(f)["videos"]

    ordered_videos = list(data.keys())
    if args.video_first_n > 0:
        ordered_videos = ordered_videos[:args.video_first_n]
        data = {video: data[video] for video in ordered_videos}

    video_list = ordered_videos
    render_video_sources = set(ordered_videos[:args.overlay_video_first_n]) if args.overlay_video_first_n > 0 else set()
    overlay_video_path_prefix = os.path.join(save_dir, "overlay_videos")
    if render_video_sources:
        ensure_dir(overlay_video_path_prefix)

    # create subprocess
    thread_num = args.num_gpus
    result_dict = mp.Manager().dict()

    processes = []
    lock = threading.Lock()

    video_num = len(video_list)
    per_thread_video_num = math.ceil(float(video_num) / float(thread_num))

    start_time = time.time()
    print('Start inference')
    for i in range(thread_num):
        if i == thread_num - 1:
            sub_video_list = video_list[i * per_thread_video_num:]
        else:
            sub_video_list = video_list[i * per_thread_video_num: (i + 1) * per_thread_video_num]
        p = mp.Process(target=sub_processor, args=(lock, i, args, data, 
                                                   output_dir, save_visualize_path_prefix,
                                                   overlay_video_path_prefix, render_video_sources,
                                                   img_folder, sub_video_list, result_dict))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    end_time = time.time()
    total_time = end_time - start_time

    result_dict = dict(result_dict)
    num_all_frames_gpus = 0
    for pid, num_all_frames in result_dict.items():
        num_all_frames_gpus += num_all_frames

    print("Total inference time: %.4f s" %(total_time))
    print("\nSave results at: {}".format(output_dir))


def sub_processor(lock, pid, args, data, save_path_prefix, save_visualize_path_prefix,
                  overlay_video_path_prefix, render_video_sources, img_folder, video_list, result_dict):
    text = 'processor %d' % pid
    with lock:
        progress = tqdm(
            total=len(video_list),
            position=pid,
            desc=text,
            ncols=0
        )
    torch.cuda.set_device(pid)

    font_size = 20
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()

    # model
    model, criterion, _ = build_model(args)  
    device = args.device
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if pid == 0:
        print('number of params:', n_parameters)

    if args.checkpoint_path is not None:
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        state_dict = checkpoint["model_state_dict"]
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(state_dict, strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        del checkpoint
    else:
        raise ValueError('Please specify the checkpoint for inference.')

    # get palette
    palette_img = os.path.join(args.dataset_path, "valid/Annotations/blackswan/00000.png")
    palette = Image.open(palette_img).getpalette()

    # start inference
    num_all_frames = 0
    model.eval()
    # 1. for each video
    for idx_, video in enumerate(video_list):
        torch.cuda.empty_cache()
        metas = []
        render_overlay_video = video in render_video_sources
        expressions = data[video]["expressions"]
        expression_list = list(expressions.keys())
        num_expressions = len(expression_list)
        video_len = len(data[video]["frames"])

        # Read all the annotation metadata
        for i in range(num_expressions):
            meta = {}
            meta["video"] = video
            meta["exp"] = expressions[expression_list[i]]["exp"]
            meta["exp_id"] = expression_list[i]  # start from 0
            meta["frames"] = data[video]["frames"]
            metas.append(meta)
        meta = metas

        # Since there are 4 annotations
        num_obj = num_expressions // 4
        # 2. for each annotator
        for anno_id in range(4):
            anno_logits = []
            anno_masks = []
            anno_boxes = []
            anno_text = []
            anno_exp_ids = []

            for obj_id in range(num_obj):
                i = obj_id * 4 + anno_id
                video_name = meta[i]["video"]
                exp = meta[i]["exp"]
                exp_id = meta[i]["exp_id"]
                frames = meta[i]["frames"]

                video_len = len(frames)

                # Load all frames of the video
                imgs = []
                origin_sizes = []
                for t in range(video_len):
                    frame = frames[t]
                    img_path = os.path.join(img_folder, video_name, frame + ".jpg")
                    img = Image.open(img_path).convert('RGB')
                    origin_w, origin_h = img.size
                    origin_sizes.append((origin_h, origin_w))
                    imgs.append(transform(img))

                imgs = torch.stack(imgs, dim=0).to(args.device)
                samples = nested_tensor_from_videos_list(imgs[None], size_divisibility=1)
                img_h, img_w = imgs.shape[-2:]
                size = torch.as_tensor([int(img_h), int(img_w)]).to(args.device)
                target = {"size": size, "video_name": video_name, "exp_idx": obj_id}

                with torch.no_grad():
                    with autocast(args.enable_amp):
                        outputs = model.infer(samples, [exp], [target])

                pred_logits = outputs["pred_logits"][0]  # [t, q, k]
                pred_masks = outputs["pred_masks"][0]  # [t, q, h, w]
                pred_boxes = outputs["pred_boxes"][0]  # [t, q, 4]

                # According to pred_logits, select the query index
                pred_scores = pred_logits.sigmoid()  # [t, q, k]
                pred_scores = pred_scores.mean(0)  # [q, K]
                max_scores, _ = pred_scores.max(-1)  # [q,]
                _, max_ind = max_scores.max(-1)  # [1,]
                max_inds = max_ind.repeat(video_len)
                pred_masks = pred_masks[range(video_len), max_inds, ...]  # [t, h, w]
                pred_masks = pred_masks.unsqueeze(0)
                pred_boxes = pred_boxes[range(video_len), max_inds]  # [t, 4]

                # Unpad and resize masks to original image sizes
                pred_masks = pred_masks[:, :, :img_h, :img_w]
                pred_masks = F.interpolate(pred_masks, size=(origin_h, origin_w), mode='bilinear', align_corners=False)
                pred_masks = pred_masks.sigmoid()[0]

                # Store the results
                pred_logits = pred_logits[range(video_len), max_inds]  # [t, k]
                anno_logits.append(pred_logits)
                anno_masks.append(pred_masks)
                anno_boxes.append(pred_boxes)
                anno_text.append(exp)
                anno_exp_ids.append(exp_id)

            # Handle a complete image (all objects of an annotator)
            anno_logits = torch.stack(anno_logits)  # [num_obj, video_len, k]
            anno_masks = torch.stack(anno_masks)  # [num_obj, video_len, h, w]
            anno_boxes = torch.stack(anno_boxes)  # [num_obj, video_len, 4]
            t, h, w = anno_masks.shape[-3:]
            anno_masks[anno_masks < 0.5] = 0.0
            background = 0.1 * torch.ones(1, t, h, w).to(args.device)
            anno_masks = torch.cat([background, anno_masks], dim=0)  # [num_obj+1, video_len, h, w]
            out_masks = torch.argmax(anno_masks, dim=0)  # int, the value indicates which object, [video_len, h, w]
            out_masks = out_masks.detach().cpu().numpy().astype(np.uint8)  # [video_len, h, w]

            anno_masks_np = None
            if args.visualize or render_overlay_video:
                anno_masks_np = (anno_masks > 0.5).detach().cpu().numpy()

            if args.visualize:
                for j, (mask, box, text) in enumerate(zip(anno_masks_np[1:], anno_boxes, anno_text)):
                    for t in range(video_len):
                        frame = frames[t]
                        img_path = os.path.join(img_folder, video_name, frame + ".jpg")
                        source_img = Image.open(img_path).convert('RGBA') # PIL image

                        color = color_list[j%len(color_list)]

                        # draw mask
                        source_img = vis_add_mask(source_img, mask[t], color)
                        draw = ImageDraw.Draw(source_img)

                        # draw boxes
                        draw_box = box[t].unsqueeze(0)
                        draw_box = rescale_bboxes(draw_box.detach(), (origin_w, origin_h)).tolist()
                        xmin, ymin, xmax, ymax = draw_box[0]
                        draw.rectangle(((xmin, ymin), (xmax, ymax)), outline=tuple(color), width=2)

                        # draw text
                        text_position = (xmin, ymin - 20)  # Adjust text position as needed
                        draw.text(text_position, text, fill=tuple(color), font=font)

                        # save
                        save_visualize_path = os.path.join(save_visualize_path_prefix, f"anno_{anno_id}", video, str(j))
                        if not os.path.exists(save_visualize_path):
                            os.makedirs(save_visualize_path)
                        source_img.save(os.path.join(save_visualize_path, '{:05d}.png'.format(t)))

            if render_overlay_video:
                for j, mask in enumerate(anno_masks_np[1:]):
                    overlay_video_path = os.path.join(
                        overlay_video_path_prefix,
                        f"anno_{anno_id}",
                        video,
                        f"exp_{anno_exp_ids[j]}.mp4",
                    )
                    write_overlay_video(
                        overlay_video_path,
                        img_folder,
                        video_name,
                        frames,
                        mask,
                        args.overlay_video_fps,
                    )

            # save results
            anno_save_path = os.path.join(save_path_prefix, f"anno_{anno_id}", video)
            if not os.path.exists(anno_save_path):
                os.makedirs(anno_save_path)
            for f in range(out_masks.shape[0]):
                img_E = Image.fromarray(out_masks[f])
                img_E.putpalette(palette)
                img_E.save(os.path.join(anno_save_path, '{:05d}.png'.format(f)))

        with lock:
            progress.update(1)

    result_dict[str(pid)] = num_all_frames
    with lock:
        progress.close()



# Post-process functions
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b.cpu() * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


# Visualization functions
def draw_reference_points(draw, reference_points, img_size, color):
    W, H = img_size
    for i, ref_point in enumerate(reference_points):
        init_x, init_y = ref_point
        x, y = W * init_x, H * init_y
        cur_color = color
        draw.line((x-10, y, x+10, y), tuple(cur_color), width=4)
        draw.line((x, y-10, x, y+10), tuple(cur_color), width=4)

def draw_sample_points(draw, sample_points, img_size, color_list):
    alpha = 255
    for i, samples in enumerate(sample_points):
        for sample in samples:
            x, y = sample
            cur_color = color_list[i % len(color_list)][::-1]
            cur_color += [alpha]
            draw.ellipse((x-2, y-2, x+2, y+2), 
                            fill=tuple(cur_color), outline=tuple(cur_color), width=1)

def vis_add_mask(img, mask, color, alpha=0.5):
    origin_img = np.asarray(img.convert('RGB')).copy().astype(np.float32)
    color = np.asarray(color, dtype=np.float32)

    mask = mask.reshape(mask.shape[0], mask.shape[1]).astype('uint8') # np
    mask = mask > 0.5

    origin_img[mask] = origin_img[mask] * (1 - alpha) + color * alpha
    origin_img = Image.fromarray(origin_img.astype(np.uint8))
    return origin_img

  

if __name__ == '__main__':
    parser = argparse.ArgumentParser('RVOS DINO: Inference')
    parser.add_argument('--config_path', '-c', required=True,
                        help='path to configuration file')
    parser.add_argument("--checkpoint_path", '-ckpt', required=True,
                        help="The checkpoint path"
                        )
    parser.add_argument("--version", default="refer_dino",
                        help="the saved ckpt and output version")
    parser.add_argument("--visualize", action='store_true')
    parser.add_argument('--num_gpus', '-ng', type=int, required=True,
                        help='number of CUDA gpus to run on. mutually exclusive with \'gpu_ids\'')
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--video_first_n", default=0, type=int, help="0 indicates use the full dataset")
    parser.add_argument("--overlay_video_first_n", default=0, type=int,
                        help="0 disables overlay video export")
    parser.add_argument("--overlay_video_fps", default=10, type=int)
    args = parser.parse_args()
    with open(args.config_path) as f:
        yaml = YAML(typ='safe', pure=True)
        config = yaml.load(f)
    config = {k: v['value'] for k, v in config.items()}
    args = {**config, **vars(args)}
    args = EasyDict(args)

    main(args)
