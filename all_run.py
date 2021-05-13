import os
import subprocess
import shutil
from glob import glob
import argparse

import cv2
import numpy as np
import skimage
from PIL import Image, ImageFont, ImageDraw
from PIL import ImageFilter
from skimage import io


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

    # for i, filepath in enumerate(sorted(glob('frames/*'))):
    #     if i > 10:
    #         os.remove(filepath)
    #         continue

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

    indexes = [
        102, 103, 104, 105 , 25, 26, 27, 28, 125, 126, 127, 128, 129, 130, 131, 132, 104, 105, 106, 107, 111, 112, 113, 114
    ]

    # Copy frames into neuronets
    for i, filepath in enumerate(sorted(glob('frames/*'))):
        if i in indexes:
            filename = filepath.split('/')[-1]
            #shutil.copy(filepath, 'vcn/images/in/' + filename)
            #shutil.copy(filepath, 'pwc/images/in/' + filename)
            shutil.copy(filepath, 'sintelall/MPI-Sintel-complete/training/frames/in/' + filename)


# Join outputs of neuronets
def load_and_caption(in_image, text):
    # Returns RGB image

    if len(in_image.shape) > 2:
        if in_image.shape[2] == 4:
            in_image = in_image[:, :, :3]
    else:
        in_image = in_image[:, :, np.newaxis]
        in_image = np.repeat(in_image, 3, 2)

    pil_img = Image.fromarray(in_image.astype(np.uint8))
    font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf", 40)
    draw = ImageDraw.Draw(pil_img)
    draw.text((0, 0), text, (255, 0, 0), font=font)
    in_image = np.array(pil_img)

    return in_image


def run():
    subp_bash('cd of-compare/raft; python run.py'
        ' --model=checkpoints/raft-things.pth --path=/content/frames')
    # subp_bash('cd of-compare/irr; python run.py')

    for i, filepath in enumerate(sorted(glob('of-compare/irr/saved_check_point/pwcnet/eval_temp/IRR_PWC/img/frames/in/*_occ.png'))):
        filename = filepath.split('/')[-1]
        shutil.move(filepath, 'out/irr/' + filename)
    for i, filepath in enumerate(sorted(glob('of-compare/raft/output/hard/*.png'))):
        filename = filepath.split('/')[-1]
        shutil.move(filepath, 'out/raft/' + filename)

    if 0:
        raft_images = sorted(glob('out13_1/raft/*.png'))
        irr_images = sorted(glob('out/irr/*.png'))
        for i in range(len(raft_images)):
            filename = raft_images[i]
            img = io.imread(filename)
            res = load_and_caption(img, 'RAFT')
            io.imsave('out/join/frame_' + str(i).zfill(4) + '.jpg', res, quality=100)

        h, w, _c = io.imread(glob('frames/*')[0]).shape
        for i in range(len(raft_images)):
            img = np.zeros((h, w * 2, 3))
            raft = io.imread(raft_images[i])
            irr = io.imread(irr_images[i])

            img[:h, :w, :] = load_and_caption(raft, 'RAFT')
            img[:h, w:, :] = load_and_caption(irr, 'IRR')
            io.imsave('out/join/frame_' + str(i).zfill(4) + '.jpg', img, quality=100)

            if 0:
                res_canvas = [0] * 4
                res_canvas[1] = img[h:h * 2, :w, :]
                res_canvas[2] = img[h:h * 2, w:w * 2, :]
                res_canvas[3] = img[:h, w:w * 2, :]




parser = argparse.ArgumentParser()
parser.add_argument('--stage', required=True, help="continue from stage: "
    "0-ffmpeg split, 1-preprocess frames, 2-run model")
#parser.add_argument('--single', action='store_true', help="launch job only once")
args = parser.parse_args()
stage = int(args.stage)

if stage <= 0:
    # Effective path must be '/'
    os.makedirs('frames', exist_ok=True)
    subp_bash('rm -rf frames/*')

    os.makedirs('out', exist_ok=True)
    subp_bash('rm -rf out/*')
    os.makedirs('out/raft', exist_ok=True)
    os.makedirs('out/irr', exist_ok=True)
    os.makedirs('out/join', exist_ok=True)

    os.makedirs('sintelall/MPI-Sintel-complete/training/frames/in', exist_ok=True)
    os.makedirs('sintelall/MPI-Sintel-complete/training/frames/out', exist_ok=True)
    subp_bash('rm -rf sintelall/MPI-Sintel-complete/training/frames/in/*')
    subp_bash('rm -rf sintelall/MPI-Sintel-complete/training/frames/out/*')

    frame_ffmpeg_split('vids/09_l4.mkv', 'frames')
if stage <= 1:
    split_frames()
if stage <= 2:
    run()