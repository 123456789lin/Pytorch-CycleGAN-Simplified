import numpy as np
import time
import os
import sys
import ntpath
from . import util
from subprocess import Popen, PIPE


if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError


def save_images(result_dir, visuals, image_path, aspect_ratio=1.0):
    """Save images to the disk.

    Parameters:
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width
    """
    image_dir = os.path.join(result_dir, 'images')
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        im = util.tensor2im(im_data)
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        util.save_image(im, save_path, aspect_ratio=aspect_ratio)


class Visualizer():
    """This class includes several functions that can display/save images and print/save logging information.
    """
    def __init__(self, opt):

        self.opt = opt  # cache the option
        self.name = opt.name
        
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)


    def display_current_results(self,img_dir, visuals, epoch):
        # save images to the disk
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        for label, image in visuals.items():
            image_numpy = util.tensor2im(image)
            img_path = os.path.join(img_dir, 'epoch%.3d_%s.png' % (epoch, label))
            util.save_image(image_numpy, img_path)


    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses):
        """print current losses on console; also save the losses to the disk
        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
        """
        message = '(epoch: %d, iters: %d) ' % (epoch, iters)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message
