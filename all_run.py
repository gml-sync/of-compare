import os
import subprocess
import shutil
from glob import glob
import argparse

import cv2
import numpy as np
import skimage


def subp_run_str(cmd, output=True):
    print('RUN:', cmd)
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)

    if output:
        for line in process.stdout:
            print(line.decode(), end='')
    rc = process.poll()
    return rc

def subp_bash(cmd):
    subp_run_str("bash -c '" + cmd + "'")




# Split frames
def frame_ffmpeg_split(video_path, output_folder):
    subp_run_str(['ffmpeg -i ' + video_path +
            ' -qscale:v 2 ' + output_folder + '/frame_%04d.jpg'])


def split_frames(stereo=False):
    print('Start frame preprocessing')
    max_area = 1240 * 436 # A Sintel-sized frame

    # Disparity. Move from frames_l and frames_r to frames
    # files_l = sorted(glob('frames_l/*'))
    # files_r = sorted(glob('frames_r/*'))
    # for i in range( 3, min(101, len(files_l)), 15 ): # 3, 8 * 24 or 100, 101
    #     target_fname_l = 'frames/frame_' + str(i * 2 + 1).zfill(4) + '.jpg'
    #     target_fname_r = 'frames/frame_' + str(i * 2 + 2).zfill(4) + '.jpg'
    #     shutil.copy(files_l[i], target_fname_l)
    #     shutil.copy(files_r[i], target_fname_r)

    # mono_list = []
    # for i in range(3, 101, 15):
    #     mono_list.append(i)
    #     mono_list.append(i + 1)

    # # Process all frames
    # for i, filepath in enumerate(sorted(glob('frames/*'))):
    #     if stereo: # Stereo
    #         if i >= 1000:
    #             os.remove(filepath)
    #             continue
    #     else: # Mono
    #         if i < 3 or i >= 102 or not i in mono_list:
    #             os.remove(filepath)
    #             continue

    img_list = sorted(glob('frames/*'))
    #bounds = find_black_frame(cv2.imread(img_list[0]))
    h, w = cv2.imread(img_list[0]).shape[:2]
    bounds = [140, 940, 0, w] # y1:y2, x1:x2

    params = []
    for i, filepath in enumerate(sorted(glob('frames/*'))):
        print(filepath)
        img = cv2.imread(filepath)
        img = img[bounds[0]:bounds[1], bounds[2]:bounds[3]] # y1:y2, x1:x2

        #img = img[::2, ::2]
        h, w = img.shape[:2]
        img = cv2.resize(img, (w//2, h//2))

        # if not(stereo and i % 2 == 1):
        #     img, params = distortion(img, params)
        # if i == 0:
        #     print('Distortion params after func:', params)

        cv2.imwrite(filepath, img, [cv2.IMWRITE_JPEG_QUALITY, 100])
        # print(Path(filepath).suffix)
        # if Path(filepath).suffix == '.jpg':
        #     imsave(filepath, img, quality=100)
        # else:
        #     imsave(filepath, img)

    # Copy frames into neuronets
    for i, filepath in enumerate(sorted(glob('frames/*'))):
        filename = filepath.split('/')[-1]
        #shutil.copy(filepath, 'vcn/images/in/' + filename)
        #shutil.copy(filepath, 'pwc/images/in/' + filename)
        shutil.copy(filepath, 'sintelall/MPI-Sintel-complete/training/frames/in/' + filename)


def run():
    # subp_bash('cd of-compare/raft; python run.py'
    #     ' --model=checkpoints/raft-things.pth --path=/content/frames')
    os.makedirs('sintelall/MPI-Sintel-complete/training/frames/in')
    os.makedirs('sintelall/MPI-Sintel-complete/training/frames/out')
    subp_bash('cd of-compare/irr; python run.py')

parser = argparse.ArgumentParser()
parser.add_argument('--stage', required=True, help="continue from stage: "
    "0-ffmpeg split, 1-preprocess frames, 2-run model")
#parser.add_argument('--single', action='store_true', help="launch job only once")
args = parser.parse_args()
stage = int(args.stage)

if stage <= 0:
    # Effective path must be '/'
    os.makedirs('frames', exist_ok=True)
    frame_ffmpeg_split('vids/11_l0.mkv', 'frames')
if stage <= 1:
    split_frames()
if stage <= 2:
    run()