#!/usr/bin/env python3
"""
AlphaPose demo – JSON ONLY.
Supports:
  - single video
  - directory of videos
  - TSV list: video_file → json_file
"""

import argparse
import os
import csv
import torch
import numpy as np
from tqdm import tqdm

from detector.apis import get_detector
from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.detector import DetectionLoader
from alphapose.utils.transforms import flip, flip_heatmap
from alphapose.utils.pPose_nms import pose_nms, write_json


# ---------------------------------------------------------------------
# JSON-only writer (isolated from AlphaPose DataWriter)
# ---------------------------------------------------------------------
class JSONOnlyWriter:
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        self.final_result = []
        self.hm_size = cfg.DATA_PRESET.HEATMAP_SIZE
        self.norm_type = cfg.LOSS.get('NORM_TYPE', None)
        self.use_heatmap_loss = (
            cfg.DATA_PRESET.get('LOSS_TYPE', 'MSELoss') == 'MSELoss'
        )

        from alphapose.utils.transforms import get_func_heatmap_to_coord
        self.heatmap_to_coord = get_func_heatmap_to_coord(cfg)
        

    def reset(self):
        self.final_result = []

    def add(self, boxes, scores, ids, hm_data, cropped_boxes, im_name):
        if boxes is None or len(boxes) == 0:
            self.final_result.append({'imgname': im_name, 'result': []})
            return

        eval_joints = list(range(hm_data.size(1)))
        pose_coords, pose_scores = [], []

        for i in range(hm_data.size(0)):
            bbox = cropped_boxes[i].tolist()

            if isinstance(self.heatmap_to_coord, list):
                # --- body & feet ---
                body_coord, body_score = self.heatmap_to_coord[0](
                    hm_data[i][eval_joints[:-110]],
                    bbox,
                    hm_shape=self.hm_size,
                    norm_type=self.norm_type
                )

                # --- face & hands ---
                face_coord, face_score = self.heatmap_to_coord[1](
                    hm_data[i][eval_joints[-110:]],
                    bbox,
                    hm_shape=self.hm_size,
                    norm_type=self.norm_type
                )

                pose_coord = np.concatenate((body_coord, face_coord), axis=0)
                pose_score = np.concatenate((body_score, face_score), axis=0)

            else:
                # fallback for non-wholebody models
                pose_coord, pose_score = self.heatmap_to_coord(
                    hm_data[i][eval_joints],
                    bbox,
                    hm_shape=self.hm_size,
                    norm_type=self.norm_type
                )

            pose_coords.append(torch.from_numpy(pose_coord).unsqueeze(0))
            pose_scores.append(torch.from_numpy(pose_score).unsqueeze(0))

        preds_img = torch.cat(pose_coords)
        preds_scores = torch.cat(pose_scores)

        if not self.args.pose_track:
            boxes, scores, ids, preds_img, preds_scores, _ = pose_nms(
                boxes, scores, ids,
                preds_img, preds_scores,
                self.args.min_box_area,
                use_heatmap_loss=self.use_heatmap_loss
            )

        result = []
        for k in range(len(scores)):
            result.append({
                'keypoints': preds_img[k],
                'kp_score': preds_scores[k],
                'proposal_score': (
                    torch.mean(preds_scores[k]) + scores[k] + 1.25 * max(preds_scores[k])
                ),
                'idx': ids[k],
                'box': [
                    boxes[k][0],
                    boxes[k][1],
                    boxes[k][2] - boxes[k][0],
                    boxes[k][3] - boxes[k][1],
                ]
            })

        self.final_result.append({
            'imgname': im_name,
            'result': result
        })

    def write(self, json_out):
        out_dir = os.path.dirname(json_out)
        os.makedirs(out_dir, exist_ok=True)

        write_json(
            self.final_result,
            out_dir,
            form=self.args.format,
            for_eval=self.args.eval,
            outputfile=os.path.basename(json_out)
        )


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def list_videos(path):
    VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
    return sorted(
        os.path.join(path, f)
        for f in os.listdir(path)
        if f.lower().endswith(VIDEO_EXTS)
    )


def read_tsv(tsv_path):
    pairs = []
    with open(tsv_path, newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        if 'video_file' not in reader.fieldnames or 'json_file' not in reader.fieldnames:
            raise ValueError(
                "TSV must contain columns: video_file, json_file"
            )
        for row in reader:
            pairs.append((row['video_file'], row['json_file']))
    return pairs


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser("AlphaPose JSON-only")

    parser.add_argument('--cfg', required=True)
    parser.add_argument('--checkpoint', required=True)

    parser.add_argument('--video', default=None,
                        help='Video file or directory (ignored if --tsv_list is used)')
    parser.add_argument('--outdir', default=None,
                        help='Output directory for JSONs (not used with --tsv_list)')

    parser.add_argument(
        "--tsv_list",
        type=str,
        default=None,
        help="TSV file with columns: video_file, json_file"
    )

    # AlphaPose args
    parser.add_argument('--detector', default="yolo")
    parser.add_argument('--pose_track', action='store_true', default=False)
    parser.add_argument('--pose_flow', action='store_true', default=False)
    parser.add_argument('--format', default="coco")
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--min_box_area', type=int, default=0)
    parser.add_argument('--gpus', default="0")
    parser.add_argument('--detbatch', type=int, default=5)
    parser.add_argument('--posebatch', type=int, default=64)
    parser.add_argument('--qsize', type=int, default=1024)
    parser.add_argument('--flip', action='store_true', default=False)
    parser.add_argument('--sp', action='store_true', default=True)

    args = parser.parse_args()
    cfg = update_config(args.cfg)

    # device
    args.gpus = [int(i) for i in args.gpus.split(',')] if torch.cuda.device_count() else [-1]
    args.device = torch.device(f"cuda:{args.gpus[0]}" if args.gpus[0] >= 0 else "cpu")
    args.tracking = args.pose_track or args.pose_flow or args.detector == 'tracker'

    # ------------------------------------------------------------------
    # Resolve (video_path, json_path) pairs
    # ------------------------------------------------------------------
    if args.tsv_list is not None:
        pairs = read_tsv(args.tsv_list)
    else:
        if args.video is None or args.outdir is None:
            raise ValueError("When --tsv_list is not used, --video and --outdir are required")

        os.makedirs(args.outdir, exist_ok=True)

        if os.path.isdir(args.video):
            videos = list_videos(args.video)
        else:
            videos = [args.video]

        pairs = []
        for v in videos:
            name = os.path.splitext(os.path.basename(v))[0]
            pairs.append((v, os.path.join(args.outdir, f"{name}.json")))

    # ------------------------------------------------------------------
    # Load models once
    # ------------------------------------------------------------------
    pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
    pose_model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    pose_model.to(args.device).eval()

    pose_dataset = builder.retrieve_dataset(cfg.DATASET.TRAIN)
    detector = get_detector(args)
    writer = JSONOnlyWriter(cfg, args)

    # ------------------------------------------------------------------
    # Process each sample
    # ------------------------------------------------------------------
    for video_path, json_out in pairs:
        print(f"\n▶ Processing video: {video_path}")
        print(f"  → Output JSON: {json_out}")

        writer.reset()

        det_loader = DetectionLoader(
            video_path,
            detector,
            cfg,
            args,
            batchSize=args.detbatch,
            mode='video',
            queueSize=args.qsize
        )
        det_loader.start()

        for _ in tqdm(
            range(det_loader.length),
            desc=os.path.basename(video_path),
            mininterval=1.0,
            miniters=10
        ):
            with torch.no_grad():
                inps, orig_img, im_name, boxes, scores, ids, cropped_boxes = det_loader.read()
                if orig_img is None:
                    break

                if boxes is None or boxes.nelement() == 0:
                    writer.add(None, None, None, None, None, im_name)
                    continue

                inps = inps.to(args.device)
                hm = pose_model(inps)

                if args.flip:
                    hm_flip = flip_heatmap(
                        hm[len(hm)//2:], pose_dataset.joint_pairs, shift=True
                    )
                    hm = (hm[:len(hm)//2] + hm_flip) / 2

                writer.add(boxes, scores, ids, hm.cpu(), cropped_boxes, im_name)

        det_loader.stop()
        writer.write(json_out)

    print("\n✅ All samples processed successfully.")


if __name__ == "__main__":
    main()
