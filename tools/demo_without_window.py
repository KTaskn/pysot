from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from operator import truediv

import os
import argparse

from tqdm import tqdm
import cv2
import torch
import numpy as np
from glob import glob

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--config', type=str, help='config file', required=True,)
parser.add_argument('--snapshot', type=str, help='model name', required=True,)
parser.add_argument('--input_video_name', default='', type=str, required=True,
                    help='videos or image files')
parser.add_argument('--output_video_name', default='', type=str, required=True,
                    help='video')
parser.add_argument('--init_rect', required=True, nargs="*", type=int, help='x1 y1 x2 y2')

args = parser.parse_args()
print(args)

assert len(args.init_rect) == 4 and [int(ent) for ent in args.init_rect]


def get_frames(video_name):
    if video_name.endswith('avi') or \
        video_name.endswith('mp4'):
        cap = cv2.VideoCapture(video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        # if images
        images = glob(os.path.join(video_name, '*.jp*'))
        images = sorted(images,
                        key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images:
            frame = cv2.imread(img)
            yield frame


def main():
    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    device = torch.device('cuda' if cfg.CUDA else 'cpu')
    print("CUDA:", device)

    # create model
    model = ModelBuilder()

    # load model
    model.load_state_dict(torch.load(args.snapshot,
        map_location=lambda storage, loc: storage.cpu()))
    model.eval().to(device)

    # build tracker
    tracker = build_tracker(model)

    first_frame = truediv
    result = []
    
    for frame in tqdm(get_frames(args.input_video_name)):
        if first_frame:
            init_rect = args.init_rect
            tracker.init(frame, init_rect)
            first_frame = False
        else:
            outputs = tracker.track(frame)
            if 'polygon' in outputs:
                polygon = np.array(outputs['polygon']).astype(np.int32)
                cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
                              True, (0, 255, 0), 3)
                mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
                mask = mask.astype(np.uint8)
                mask = np.stack([mask, mask*255, mask]).transpose(1, 2, 0)
                frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
            else:
                bbox = list(map(int, outputs['bbox']))
                cv2.rectangle(frame, (bbox[0], bbox[1]),
                              (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                              (0, 255, 0), 3)
            result.append(frame)
    
    fmt = cv2.VideoWriter_fourcc(*'mp4v') # ファイル形式(ここではmp4)
    frame_rate = 20.0
    h, w = result[0].shape[:2]
    writer = cv2.VideoWriter(args.output_video_name, fmt, frame_rate, (w, h))
    for frame in result:
        writer.write(frame)
    writer.release()

if __name__ == '__main__':
    main()
