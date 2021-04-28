import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

from utils.checkpoints import checkpoint_load_path, checkpoint_save_path, save_model_txt, load_model_txt
from skimage import io

print('Im here!')

DEVICE = 'cpu'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def demo(args):
    save_dir = 'output'
    os.makedirs(save_dir, exist_ok=True)

    model = torch.nn.DataParallel(RAFT(args))
    #load_model_txt(model, 'checkpoints/raft-things.pth')
    model.load_state_dict(torch.load(args.model,  map_location=torch.device('cpu')))

    model = model.module
    model.to(DEVICE)
    model.eval()
    print('Model loaded')

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images = sorted(images)
        for imfile1, imfile2 in zip(images[:-1], images[1:]): # Genius!
            print('Processing', imfile1)
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)










            flow_seq, occ_seq = model(image1, image2, iters=20, test_mode=True) # b c h w
            flow = flow_seq[-1][0] # last prediction in sequence + first item in batch
            occ = occ_seq[-1][0]
            flow = padder.unpad(flow).cpu() # c h w
            occ = padder.unpad(occ).cpu()

            # shape=(h, w) bool
            occ = occ[0].numpy()
            # shape=(h, w) float32 in [0, 1]

            path = save_dir
            io.imsave(path / '{}.png'.format(imfile1), occ)
            
            # f = flow.permute(1,2,0).numpy()
            # flow_img = flow_viz.flow_to_image(f)
            # io.imsave(path / '{:04d}_flow.png'.format(val_id), flow_img)

            # f = flow_gt.permute(1,2,0).cpu().numpy()






            # flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            # viz(image1, flow_up)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    print('Starting RAFT')
    demo(args)
