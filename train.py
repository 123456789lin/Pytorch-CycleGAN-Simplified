# python train.py --dataroot ./datasets/horse2zebra --name horse2zebra_cyclegan --model cycle_gan
# python test.py --dataroot ./datasets/horse2zebra --name horse2zebra_cyclegan --model cycle_gan


import os
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer



if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options

    model = create_model(opt)      # create a model given opt.model and other options
    start_epoch = model.setup(opt)               # regular setup: load and print networks; create schedulers
    
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations
    
    img_dir = os.path.join(opt.checkpoints_dir, opt.name, 'images')

    for epoch in range(start_epoch, opt.n_epochs + opt.n_epochs_decay + 1):

        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        
        for i, data in enumerate(dataset):  # inner loop within one epoch

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                model.compute_visuals()
                visualizer.display_current_results(img_dir,model.get_current_visuals(), epoch)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                visualizer.print_current_losses(epoch, epoch_iter, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d' % (epoch, opt.n_epochs + opt.n_epochs_decay))
