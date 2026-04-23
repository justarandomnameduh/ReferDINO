import argparse
import os
import warnings
from pathlib import Path

import cv2
import imageio.v3 as iio
import matplotlib.colors
import misc as utils
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as Func
from easydict import EasyDict
from models import build_model
from PIL import Image, ImageDraw
from ruamel.yaml import YAML
from torch.cuda.amp import autocast
from torchvision.io import read_video

from misc import nested_tensor_from_videos_list

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

transform = T.Compose(
    [
        T.Resize(360),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def ensure_dir(path: str | None) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def collect_input_frames(input_path: str, frame_step: int, fallback_fps: float) -> tuple[list[Image.Image], list[str], float, str]:
    if os.path.isdir(input_path):
        frame_paths = sorted(
            path
            for path in Path(input_path).iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        )
        if not frame_paths:
            raise FileNotFoundError(f"No image frames found in directory: {input_path}")
        frames = [Image.open(path).convert("RGB") for path in frame_paths]
        frame_names = [path.stem for path in frame_paths]
        return frames, frame_names, fallback_fps, Path(input_path).name

    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)

    video_frames, _, info = read_video(input_path, pts_unit="sec")
    frames = []
    frame_names = []
    source_name = Path(input_path).name
    for index in range(0, len(video_frames), frame_step):
        source_frame = Func.to_pil_image(video_frames[index].permute(2, 0, 1))
        frames.append(source_frame.convert("RGB"))
        frame_names.append(f"frame_{len(frame_names):05d}")

    fps = float(info.get("video_fps") or 0.0)
    if fps <= 0:
        fps = fallback_fps
    else:
        fps = fps / frame_step
    return frames, frame_names, fps, source_name


def main(args):
    model = load_model(args)
    infer(model, args.video, args.text, args)


def infer(model, video, text, args):
    exp = " ".join(text.lower().split())
    frames, frame_names, fps, input_name = collect_input_frames(video, args.frame_step, args.video_fps)

    save_name = args.save_name if args.save_name is not None else input_name
    if not save_name.endswith(".mp4"):
        save_name += ".mp4"

    output_dir = args.output_dir
    ensure_dir(output_dir)
    ensure_dir(args.overlay_frames_dir)
    ensure_dir(args.pred_masks_dir)
    save_video = os.path.join(output_dir, save_name)

    video_len = len(frames)
    imgs = []
    origin_w = 0
    origin_h = 0
    for img in frames:
        origin_w, origin_h = img.size
        imgs.append(transform(img))

    imgs = torch.stack(imgs, dim=0).to(args.device)
    samples = nested_tensor_from_videos_list(imgs[None], size_divisibility=1)
    img_h, img_w = imgs.shape[-2:]
    size = torch.as_tensor([int(img_h), int(img_w)]).to(args.device)
    target = {"size": size}

    print("begin inference")
    with torch.no_grad():
        with autocast(args.enable_amp):
            outputs = model.infer(samples, [exp], [target])

    pred_logits = outputs["pred_logits"][0]
    pred_masks = outputs["pred_masks"][0]
    pred_boxes = outputs["pred_boxes"][0]

    pred_scores = pred_logits.sigmoid()
    pred_scores = pred_scores.mean(0)
    max_scores, _ = pred_scores.max(-1)
    _, max_ind = max_scores.max(-1)
    max_inds = max_ind.repeat(video_len)
    pred_masks = pred_masks[range(video_len), max_inds, ...]
    pred_masks = pred_masks.unsqueeze(0)
    pred_boxes = pred_boxes[range(video_len), max_inds].cpu().numpy()

    pred_masks = pred_masks[:, :, :img_h, :img_w].cpu()
    pred_masks = F.interpolate(pred_masks, size=(origin_h, origin_w), mode="bilinear", align_corners=False)
    pred_masks = (pred_masks.sigmoid() > 0.5).squeeze(0).cpu().numpy()

    print("saving")

    color = "#DC143C"
    color = (np.array(matplotlib.colors.hex2color(color)) * 255).astype("uint8")

    save_imgs = []
    for index, img in enumerate(frames):
        rendered = vis_add_mask(img, pred_masks[index], color, args.mask_edge_width)
        draw = ImageDraw.Draw(rendered)
        draw_boxes = pred_boxes[index][None]
        draw_boxes = rescale_bboxes(draw_boxes, (origin_w, origin_h)).tolist()

        if args.show_box:
            xmin, ymin, xmax, ymax = draw_boxes[0]
            draw.rectangle(((xmin, ymin), (xmax, ymax)), outline=tuple(color), width=5)

        if args.save_images:
            rendered.save(os.path.join(output_dir, f"{index:05d}.png"))

        if args.overlay_frames_dir:
            rendered.save(os.path.join(args.overlay_frames_dir, f"{frame_names[index]}.png"))

        if args.pred_masks_dir:
            mask_image = Image.fromarray(pred_masks[index].astype(np.uint8) * 255).convert("L")
            mask_image.save(os.path.join(args.pred_masks_dir, f"{frame_names[index]}.png"))

        save_imgs.append(np.asarray(rendered).copy())

    if not args.skip_video_write:
        iio.imwrite(save_video, save_imgs, fps=fps)
        print(f"result video saved to {save_video}")


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    b = np.stack([x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h], axis=1)
    return b


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * np.array([img_w, img_h, img_w, img_h], dtype=np.float32)
    return b


def vis_add_mask(img, mask, color, edge_width=3):
    origin_img = np.asarray(img.convert("RGB")).copy()
    color = np.array(color)

    mask = mask.reshape(mask.shape[0], mask.shape[1]).astype("uint8")
    mask = mask > 0.5

    kernel = np.ones((edge_width, edge_width), np.uint8)
    mask_dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    edge_mask = mask_dilated & ~mask

    origin_img[mask] = origin_img[mask] * 0.5 + color * 0.5
    origin_img[edge_mask] = color
    return Image.fromarray(origin_img)


def load_model(args):
    model, _, _ = build_model(args)
    model.to(args.device)
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser("RVOS DINO: Inference")
    parser.add_argument("--config_path", default="configs/ytvos_swinb.yaml", help="path to configuration file")
    parser.add_argument("--checkpoint_path", "-ckpt", required=True, help="The checkpoint path")
    parser.add_argument("--frame_step", default=1, type=int, help="Sampling interval of the video")
    parser.add_argument("--output_dir", default="output/demo")
    parser.add_argument("--video", required=True, help="Path to an input video or a directory of frames")
    parser.add_argument("--text", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--show_box", action="store_true")
    parser.add_argument("--mask_edge_width", default=6, type=int)
    parser.add_argument("--bar_height", default=80, type=int)
    parser.add_argument("--font_size", default=60, type=int)
    parser.add_argument("--save_name", type=str, default=None)
    parser.add_argument("--tracking_alpha", default=0.1, type=float)
    parser.add_argument("--overlay_frames_dir", type=str, default=None)
    parser.add_argument("--pred_masks_dir", type=str, default=None)
    parser.add_argument("--skip_video_write", action="store_true")
    parser.add_argument("--video_fps", default=10.0, type=float)
    args = parser.parse_args()

    with open(args.config_path) as file_handle:
        yaml = YAML(typ="safe", pure=True)
        config = yaml.load(file_handle)
    config = {key: value["value"] for key, value in config.items()}
    args = {**config, **vars(args)}
    args = EasyDict(args)
    args.GroundingDINO.tracking_alpha = args.tracking_alpha
    main(args)
