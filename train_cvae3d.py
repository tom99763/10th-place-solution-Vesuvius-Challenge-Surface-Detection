from src.models.CVAE3d import *
from src.trainers.losses import *
from src.models.utils import *
import logging
import colorama
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
import random
import os
from src.utils import logger, tools, saver, summaries
import logging
import colorama
from src.datasets.scroll_dataset_cvae3d import *
import json


clear = colorama.Style.RESET_ALL
blue = colorama.Fore.CYAN + colorama.Style.BRIGHT
green = colorama.Fore.GREEN + colorama.Style.BRIGHT
magenta = colorama.Fore.MAGENTA + colorama.Style.BRIGHT


'''pipeline explain 
1. train VAEGAN from coarse to fine
2. when current scale is larger than opt.vae_scale, then start to use discriminator
3. there's no downsample in discriminator
'''


def get_parser():
    parser = argparse.ArgumentParser()

    # load, input, save configurations:
    parser.add_argument('--netG', default='', help='path to netG (to continue training)')
    parser.add_argument('--netD', default='', help='path to netD (to continue training)')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--validation_split', default='./splits_final.json', type=str, required=True, help='manual seed')

    # networks hyper parameters:
    parser.add_argument('--nc-im', type=int, default=1, help='# channels')
    parser.add_argument('--nfc', type=int, default=64, help='model basic # channels')
    parser.add_argument('--latent-dim', type=int, default=128, help='Latent dim size')
    parser.add_argument('--vae-levels', type=int, default=3, help='# VAE levels')
    parser.add_argument('--enc-blocks', type=int, default=2, help='# encoder blocks')
    parser.add_argument('--ker-size', type=int, default=3, help='kernel size')
    parser.add_argument('--num-layer', type=int, default=5, help='number of layers')
    parser.add_argument('--stride', default=1, help='stride')
    parser.add_argument('--padd-size', type=int, default=1, help='net pad size')
    parser.add_argument('--generator', type=str, default='GeneratorHPVAEGAN', help='generator model')
    parser.add_argument('--discriminator', type=str, default='WDiscriminator3D', help='discriminator model')

    # pyramid parameters:
    parser.add_argument('--scale-factor', type=float, default=0.75, help='pyramid scale factor')
    parser.add_argument('--noise_amp', type=float, default=0.1, help='addative noise cont weight')
    parser.add_argument('--min-size', type=int, default=32, help='image minimal size at the coarser scale')
    parser.add_argument('--max-size', type=int, default=256, help='image minimal size at the coarser scale')

    # optimization hyper parameters:
    parser.add_argument('--niter', type=int, default=50000, help='number of iterations to train per scale')
    parser.add_argument('--lr-g', type=float, default=0.0005, help='learning rate, default=0.0005')
    parser.add_argument('--lr-d', type=float, default=0.0005, help='learning rate, default=0.0005')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--lambda-grad', type=float, default=0.1, help='gradient penelty weight')
    parser.add_argument('--rec-weight', type=float, default=10., help='reconstruction loss weight')
    parser.add_argument('--kl-weight', type=float, default=1., help='reconstruction loss weight')
    parser.add_argument('--disc-loss-weight', type=float, default=1.0, help='discriminator weight')
    parser.add_argument('--lr-scale', type=float, default=0.2, help='scaling of learning rate for lower stages')
    parser.add_argument('--train-depth', type=int, default=1, help='how many layers are trained if growing')
    parser.add_argument('--grad-clip', type=float, default=5, help='gradient clip')
    parser.add_argument('--const-amp', action='store_true', default=False, help='constant noise amplitude')
    parser.add_argument('--train-all', action='store_true', default=False, help='train all levels w.r.t. train-depth')

    # Dataset
    parser.add_argument('--video-path', required=True, help='video path')
    parser.add_argument('--start-frame', default=0, type=int, help='start frame number')
    parser.add_argument('--max-frames', default=13, type=int, help='# frames to save')
    parser.add_argument('--hflip', action='store_true', default=False, help='horizontal flip')
    parser.add_argument('--img-size', type=int, default=256)
    parser.add_argument('--sampling-rates', type=int, nargs='+', default=[4, 3, 2, 1], help='sampling rates')
    parser.add_argument('--stop-scale-time', type=int, default=-1)
    parser.add_argument('--data-rep', type=int, default=1000, help='data repetition')

    # main arguments
    parser.add_argument('--checkname', type=str, default='DEBUG', help='check name')
    parser.add_argument('--mode', default='train', help='task to be done')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--print-interval', type=int, default=100, help='print interva')
    parser.add_argument('--visualize', action='store_true', default=False, help='visualize using tensorboard')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables cuda')

    parser.set_defaults(hflip=False)
    opt = parser.parse_args()
    return opt


def train(opt, netG):
    pass

def prepare_dataset(opt):
    pass

def setup_parameters(opt):
    # utilities
    opt.saver = saver.VideoSaver(opt)
    opt.summary = summaries.TensorboardSummary(opt.saver.experiment_dir)
    device = 'cuda' if torch.cuda.is_available() and not opt.no_cuda else 'cpu'
    opt.device = device

    # Initial config
    opt.noise_amp_init = opt.noise_amp
    opt.scale_factor_init = opt.scale_factor

    # Adjust scales
    adjust_scales2image(opt.img_size, opt)

    # Manual seed
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    logging.info("Random Seed: {}".format(opt.manualSeed))
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    # Reconstruction loss
    opt.rec_loss = torch.nn.MSELoss()

    # Initial parameters
    opt.scale_idx = 0
    opt.nfc_prev = 0
    opt.Noise_Amps = []

    #get validation split
    with open(opt.validation_split, 'r') as f:
        splits = json.load(f)
    opt.train_val_splits_ids = splits


def main():
    opt = get_parser()
    assert opt.vae_levels > 0
    assert opt.disc_loss_weight > 0

    setup_parameters(opt)

    for fold_id in range(len(opt.train_val_splits_ids)):
        print(f"****** Fold: {fold_id} starts ******")
        opt.fold_id = fold_id
        prepare_dataset(opt)

        #show loggings
        with open(os.path.join(opt.saver.experiment_dir, 'args.txt'), 'w') as args_file:
            for argument, value in sorted(vars(opt).items()):
                if type(value) in (str, int, float, tuple, list, bool):
                    args_file.write('{}: {}\n'.format(argument, value))

        with logger.LoggingBlock("Commandline Arguments", emph=True):
            for argument, value in sorted(vars(opt).items()):
                if type(value) in (str, int, float, tuple, list):
                    logging.info('{}: {}'.format(argument, value))

        with logger.LoggingBlock("Experiment Summary", emph=True):
            video_file_name, checkname, experiment = opt.saver.experiment_dir.split('/')[-3:]
            logging.info("{}Video file :{} {}{}".format(magenta, clear, video_file_name, clear))
            logging.info("{}Checkname  :{} {}{}".format(magenta, clear, checkname, clear))
            logging.info("{}Experiment :{} {}{}".format(magenta, clear, experiment, clear))

            with logger.LoggingBlock("Commandline Summary", emph=True):
                logging.info("{}Start frame    :{} {}{}".format(blue, clear, opt.start_frame, clear))
                logging.info("{}Max frames     :{} {}{}".format(blue, clear, opt.max_frames, clear))
                logging.info("{}Generator      :{} {}{}".format(blue, clear, opt.generator, clear))
                logging.info("{}Iterations     :{} {}{}".format(blue, clear, opt.niter, clear))
                logging.info("{}Rec. Weight    :{} {}{}".format(blue, clear, opt.rec_weight, clear))
                logging.info("{}Sampling rates :{} {}{}".format(blue, clear, opt.sampling_rates, clear))


        netG = GeneratorHPVAEGAN(opt).to(opt.device)
        if opt.netG != '':
            if not os.path.isfile(opt.netG):
                raise RuntimeError("=> no <G> checkpoint found at '{}'".format(opt.netG))
            checkpoint = torch.load(opt.netG)
            opt.scale_idx = checkpoint['scale']
            opt.resumed_idx = checkpoint['scale']
            opt.resume_dir = '/'.join(opt.netG.split('/')[:-1])
            for _ in range(opt.scale_idx):
                netG.init_next_stage()
            netG.load_state_dict(checkpoint['state_dict'])
            opt.Noise_Amps = torch.load(os.path.join(opt.resume_dir, 'Noise_Amps.pth'))['data']
        else:
            opt.resumed_idx = -1

        while opt.scale_idx < opt.stop_scale + 1:
            if (opt.scale_idx > 0) and (opt.resumed_idx != opt.scale_idx):
                netG.init_next_stage()

            #train current scale
            train(opt, netG)

            # Increase scale
            opt.scale_idx += 1





if __name__ == '__main__':
    pass



