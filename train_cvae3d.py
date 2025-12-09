from src.models.CVAE3d import *
from src.procs.proc_utils import *
from src.trainers.losses import *
import argparse
import random
import os
#from src.utils import logger, saver, summaries, tools
from src.utils.logger import  *
from src.utils.saver import *
from src.utils.summaries import *
from src.utils.tools import *
import colorama
from src.datasets.scroll_dataset_cvae3d import *
import json
import torch.optim as optim
from tqdm import tqdm
from monai.metrics import DiceMetric


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

    # files
    parser.add_argument('--dataset_dir', default='./data/vesuvius-challenge-surface-detection', type=str,
                        help='dataset directory')
    parser.add_argument('--oof_dir', default='./nnunet/nnUNet_results/Dataset900_VesuviusScroll'
                                                 '/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/'
                                                 'oof', type=str,
                        help='out-of-fold directory')
    parser.add_argument('--validation_split', default='./splits_final.json', type=str, help='validation split')

    # networks hyper parameters:
    parser.add_argument('--nc-im', type=int, default=1, help='# channels')
    parser.add_argument('--nfc', type=int, default=64, help='model basic # channels')
    parser.add_argument('--latent-dim', type=int, default=128, help='Latent dim size')
    parser.add_argument('--vae-levels', type=int, default=2, help='Determine # layers being used for VAE')
    parser.add_argument('--enc-blocks', type=int, default=2, help='# encoder blocks')
    parser.add_argument('--ker-size', type=int, default=3, help='kernel size')
    parser.add_argument('--num-layer', type=int, default=5, help='number of layers')
    parser.add_argument('--stride', default=1, help='stride')
    parser.add_argument('--padd-size', type=int, default=1, help='net pad size')
    parser.add_argument('--generator', type=str, default='GeneratorHPVAEGAN', help='generator model')
    parser.add_argument('--discriminator', type=str, default='WDiscriminator3D', help='discriminator model')

    # pyramid parameters:
    parser.add_argument('--scale-factor', type=float, default=0.5, help='pyramid scale factor')
    parser.add_argument('--noise_amp', type=float, default=0.1, help='addative noise cont weight')
    parser.add_argument('--min-size', type=int, default=32, help='image minimal size at the coarser scale')
    parser.add_argument('--max-size', type=int, default=256, help='image minimal size at the coarser scale')

    # optimization hyper parameters:
    parser.add_argument('--niter', type=int, default=40, help='number of iterations to train per scale')
    parser.add_argument('--lr-g', type=float, default=0.0005, help='learning rate, default=0.0005')
    parser.add_argument('--lr-d', type=float, default=0.0005, help='learning rate, default=0.0005')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--lambda-grad', type=float, default=0.1, help='gradient penelty weight')
    parser.add_argument('--rec-weight', type=float, default=1., help='reconstruction loss weight')
    parser.add_argument('--kl-weight', type=float, default=1., help='reconstruction loss weight')
    parser.add_argument('--disc-loss-weight', type=float, default=1.0, help='discriminator weight')
    parser.add_argument('--lr-scale', type=float, default=0.2, help='scaling of learning rate for lower stages')
    parser.add_argument('--train-depth', type=int, default=1, help='how many layers are trained if growing')
    parser.add_argument('--grad-clip', type=float, default=10, help='gradient clip')
    parser.add_argument('--const-amp', action='store_true', default=False, help='constant noise amplitude')
    parser.add_argument('--train-all', action='store_true', default=False, help='train all levels w.r.t. train-depth')

    # Dataset
    parser.add_argument('--experiment-dir', default='./cvae3d', help='video path')
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
    parser.add_argument('--batch-size', type=int, default=4, help='batch size')
    parser.add_argument('--print-interval', type=int, default=100, help='print interva')
    parser.add_argument('--visualize', action='store_true', default=False, help='visualize using tensorboard')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables cuda')

    parser.set_defaults(hflip=False)
    opt = parser.parse_args()
    return opt


def train(opt, netG):
    with LoggingBlock("Updating dataset", emph=True):
        logging.info(f'training scale :{opt.scale_idx} at fold {opt.selected_fold}')
        prepare_dataset(opt)
        opt.train_dataset.setup_scale(opt.scale_idx)
        opt.val_dataset.setup_scale(opt.scale_idx)

    if not hasattr(opt, 'Z_init_size'):
        initial_size = get_scales_by_index(0, opt.scale_factor, opt.stop_scale, opt.img_size)
        initial_size = [initial_size] * 3
        opt.Z_init_size = [opt.batch_size, opt.latent_dim, *initial_size]

    if opt.vae_levels < opt.scale_idx + 1:
        D_curr = WDiscriminator3D(opt).to(opt.device)
        if (opt.netG != '') and (opt.resumed_idx == opt.scale_idx):
            D_curr.load_state_dict(
                torch.load('{}/netD_{}_{}.pth'.format(opt.resume_dir, opt.scale_idx - 1, opt.selected_fold), map_location=opt.device)['state_dict'])
        elif opt.vae_levels < opt.scale_idx:
            D_curr.load_state_dict(
                torch.load('{}/netD_{}_{}.pth'.format(opt.saver.experiment_dir, opt.scale_idx - 1, opt.selected_fold), map_location=opt.device)['state_dict'])
        optimizerD = optim.Adam(D_curr.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))

    parameter_list = []
    if not opt.train_all:
        if opt.vae_levels < opt.scale_idx + 1:
            train_depth = min(opt.train_depth, len(netG.body) - opt.vae_levels + 1)
            parameter_list += [
                {"params": block.parameters(),
                 "lr": opt.lr_g * (opt.lr_scale ** (len(netG.body[-train_depth:]) - 1 - idx))}
                for idx, block in enumerate(netG.body[-train_depth:])]
        else:
            # VAE
            parameter_list += [{"params": netG.encode.parameters(), "lr": opt.lr_g * (opt.lr_scale ** opt.scale_idx)},
                               {"params": netG.decoder.parameters(), "lr": opt.lr_g * (opt.lr_scale ** opt.scale_idx)}]
            parameter_list += [
                {"params": block.parameters(),
                 "lr": opt.lr_g * (opt.lr_scale ** (len(netG.body[-opt.train_depth:]) - 1 - idx))}
                for idx, block in enumerate(netG.body[-opt.train_depth:])]
    else:
        if len(netG.body) < opt.train_depth:
            parameter_list += [{"params": netG.encode.parameters(), "lr": opt.lr_g * (opt.lr_scale ** opt.scale_idx)},
                               {"params": netG.decoder.parameters(), "lr": opt.lr_g * (opt.lr_scale ** opt.scale_idx)}]
            parameter_list += [
                {"params": block.parameters(),
                 "lr": opt.lr_g * (opt.lr_scale ** (len(netG.body) - 1 - idx))}
                for idx, block in enumerate(netG.body)]
        else:
            parameter_list += [
                {"params": block.parameters(),
                 "lr": opt.lr_g * (opt.lr_scale ** (len(netG.body[-opt.train_depth:]) - 1 - idx))}
                for idx, block in enumerate(netG.body[-opt.train_depth:])]

    optimizerG = optim.Adam(parameter_list, lr=opt.lr_g, betas=(opt.beta1, 0.999))

    # Parallel
    if opt.device == 'cuda':
        G_curr = torch.nn.DataParallel(netG)
        if opt.vae_levels < opt.scale_idx + 1:
            D_curr = torch.nn.DataParallel(D_curr)
    else:
        G_curr = netG

    progressbar_args = {
        "iterable": range(opt.niter * 2) if opt.vae_levels < opt.scale_idx + 1 else range(opt.niter),
        "desc": "Training scale [{}/{}]".format(opt.scale_idx + 1, opt.stop_scale + 1),
        "train": True,
        "offset": 0,
        "logging_on_update": False,
        "logging_on_close": True,
        "postfix": True
    }
    epoch_iterator = tools.create_progressbar(**progressbar_args)

    val_topo_losses = []
    val_surf_losses = []
    val_dice_metric = DiceMetric(include_background=False, reduction="mean", ignore_empty=True)
    val_num_samples = 0
    best_score = 0

    for iteration in epoch_iterator:
        ############################
        # train step
        ###########################
        looper = tqdm(opt.train_loader)
        G_curr.train()
        if opt.vae_levels < opt.scale_idx + 1:
            D_curr.train()
        for mask_gt, mask_gt_0, mask_pred, mask_pred_0 in looper:
            mask_gt = mask_gt.to(opt.device)
            mask_gt_0 = mask_gt_0.to(opt.device)
            noise_init = generate_noise(size=opt.Z_init_size, device=opt.device)

            ############################
            # calculate noise_amp
            ###########################
            if iteration == 0:
                if opt.const_amp:
                    opt.Noise_Amps.append(1)
                else:
                    with torch.no_grad():
                        if opt.scale_idx == 0:
                            opt.noise_amp = 1
                            opt.Noise_Amps.append(opt.noise_amp)
                        else:
                            opt.Noise_Amps.append(0)
                            G_curr.to(opt.device)
                            z_reconstruction, _, _ = G_curr(video = mask_gt_0, noise_amp = opt.Noise_Amps, mode="rec")
                            RMSE = torch.sqrt(F.mse_loss(mask_gt, z_reconstruction))
                            opt.noise_amp = opt.noise_amp_init * RMSE.item() / opt.batch_size
                            opt.Noise_Amps[-1] = opt.noise_amp

            ############################
            # (1) Update VAE network
            ###########################
            total_loss = 0

            generated, generated_vae, (mu, logvar) = G_curr(video = mask_gt_0, noise_amp = opt.Noise_Amps, mode="rec")

            if opt.vae_levels >= opt.scale_idx + 1:
                rec_vae_loss = opt.rec_loss(generated, mask_gt) + opt.rec_loss(generated_vae, mask_gt_0)
                rec_vae_loss += 0.5 *(opt.topo_loss(generated, mask_gt) + opt.soft_surf_loss(generated, mask_gt) + \
                               opt.topo_loss(generated_vae, mask_gt_0) + opt.soft_surf_loss(generated_vae, mask_gt_0))

                kl_loss = kl_criterion(mu, logvar)
                vae_loss = opt.rec_weight * rec_vae_loss + opt.kl_weight * kl_loss

                total_loss += vae_loss
                looper.set_postfix(loss=f"total_loss: {total_loss} -- vae_loss: {rec_vae_loss} -- kl_loss :{kl_loss}")
            else:
                ############################
                # (2) Update D network: maximize D(x) + D(G(z))
                ###########################
                # train with real
                #################

                # Train 3D Discriminator
                D_curr.zero_grad()
                output = D_curr(mask_gt)
                errD_real = -output.mean()

                # train with fake
                #################
                fake, _ = G_curr(video = noise_init, noise_amp =  opt.Noise_Amps, noise_init=noise_init, mode="rand")

                # Train 3D Discriminator
                output = D_curr(fake.detach())
                errD_fake = output.mean()

                gradient_penalty = calc_gradient_penalty(D_curr, mask_gt, fake, opt.lambda_grad,
                                                         opt.device)
                errD_total = errD_real + errD_fake + gradient_penalty
                errD_total.backward()
                optimizerD.step()

                ############################
                # (3) Update G network: maximize D(G(z))
                ###########################
                errG_total = 0
                rec_loss = opt.rec_loss(generated, mask_gt)
                rec_loss += 0.5 * (opt.topo_loss(generated, mask_gt) + opt.soft_surf_loss(generated, mask_gt) + \
                               opt.topo_loss(generated_vae, mask_gt_0) + opt.soft_surf_loss(generated_vae, mask_gt_0))
                errG_total += opt.rec_weight * rec_loss

                # Train with 3D Discriminator
                output = D_curr(fake)
                errG = -output.mean() * opt.disc_loss_weight
                errG_total += errG

                total_loss += errG_total

                looper.set_postfix(loss=f"total_loss: {total_loss} -- errG: {errG} -- errD :{errD_total} -- gp:{gradient_penalty}")

            G_curr.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(G_curr.parameters(), opt.grad_clip)
            optimizerG.step()

        # Update progress bar
        epoch_iterator.set_description('Scale [{}/{}], Iteration [{}/{}]'.format(
            opt.scale_idx + 1, opt.stop_scale + 1,
            iteration + 1, opt.niter,
        ))

        ############################
        # val step
        ###########################
        G_curr.eval()
        if opt.vae_levels < opt.scale_idx + 1:
            D_curr.eval()
        with torch.no_grad():
            for mask_gt, mask_gt_0, mask_pred, mask_pred_0 in tqdm(opt.val_loader):
                mask_gt = mask_gt.to(opt.device)
                # mask_gt_0 = mask_gt_0.to(opt.device)
                # mask_pred = mask_pred.to(opt.device)
                mask_pred_0 = mask_pred_0.to(opt.device)

                #forward
                mask_pred_rec, _, _ = G_curr(video = mask_pred_0, noise_amp = opt.Noise_Amps, mode="rec")
                mask_pred_rec = mask_pred_rec.clamp(0, 1)
                ignore_mask = (mask_gt != 2).float()
                target_mask = mask_gt * ignore_mask
                mask_pred_bin = (mask_pred_rec > opt.threshold).float() * ignore_mask

                topo_loss = opt.topo_loss(mask_pred_rec, target_mask)  # scalar
                surf_loss = opt.surf_loss(mask_pred_rec, target_mask)  # scalar

                val_dice_metric(y_pred=mask_pred_bin, y=(target_mask).long())
                val_topo_losses.append(topo_loss * mask_gt.shape[0])
                val_surf_losses.append(surf_loss * mask_gt.shape[0])
                val_num_samples += mask_gt.shape[0]

            avg_topo_loss = torch.stack(val_topo_losses).sum() / val_num_samples
            avg_surf_loss = torch.stack(val_surf_losses).sum() / val_num_samples

            # Convert to metric scores (higher = better)
            topo_score = 1.0 - avg_topo_loss
            surf_score = 1.0 - avg_surf_loss

            # Final Dice from MONAI
            dice_score = val_dice_metric.aggregate().mean().item()

            # === Competition metric ===
            #comp_metric = 0.30 * topo_score + 0.35 * surf_score + 0.35 * dice_score
            comp_metric = 0.40 * topo_score + 0.60 * surf_score #+ 0.35 * dice_score

            print(f"\nVAL Epoch {iteration} │ "
                  f"Dice: {dice_score:.4f} │ "
                  f"Topo: {topo_score:.4f} │ "
                  f"Surf: {surf_score:.4f} │ "
                  f"→ COMP: {comp_metric:.4f} ←\n")

            # === Reset everything ===
            val_topo_losses = []
            val_surf_losses = []
            val_dice_metric.reset()
            val_num_samples = 0

        if comp_metric > best_score:
            best_score = comp_metric
            opt.saver.save_checkpoint({'data': opt.Noise_Amps}, 'Noise_Amps.pth')
            opt.saver.save_checkpoint({
                'scale': opt.scale_idx,
                'state_dict': netG.state_dict(),
                'optimizer': optimizerG.state_dict(),
                'noise_amps': opt.Noise_Amps,
            }, 'netG_{}_{}_ {:.3f}_{:.3f}_{:.3f}_{:.3f}.pth'.format(opt.selected_fold, opt.scale_idx,
                                                                    topo_score, surf_score, dice_score, comp_metric))
            if opt.vae_levels < opt.scale_idx + 1:
                opt.saver.save_checkpoint({
                    'scale': opt.scale_idx,
                    'state_dict': D_curr.module.state_dict() if opt.device == 'cuda' else D_curr.state_dict(),
                    'optimizer': optimizerD.state_dict(),
                }, 'netD_{}_{}.pth'.format(opt.scale_idx, opt.selected_fold))


def prepare_dataset(opt):
    train_dataset = CVAEDataset(opt, True)
    val_dataset = CVAEDataset(opt, False)
    opt.train_dataset = train_dataset
    opt.val_dataset = val_dataset
    opt.train_loader = DataLoader(opt.train_dataset,
                                  shuffle=True,
                                  drop_last=False,
                                  batch_size=opt.batch_size,
                                  num_workers=4,
                                  persistent_workers=True
                                  )
    opt.val_loader = DataLoader(opt.val_dataset,
                                shuffle=False,
                                drop_last=False,
                                batch_size=opt.batch_size,
                                num_workers=4,
                                persistent_workers = True
                                )


def setup_parameters(opt):
    # utilities
    opt.saver = VideoSaver(opt)
    #opt.summary = summaries.TensorboardSummary(opt.saver.experiment_dir)
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

    # losses
    opt.rec_loss = torch.nn.MSELoss()
    opt.topo_loss = FastClDiceLoss()
    opt.soft_surf_loss = SoftSDFLoss()
    opt.surf_loss = SurfaceLoss()

    # Initial parameters
    opt.scale_idx = 0
    opt.nfc_prev = 0
    opt.Noise_Amps = []
    opt.threshold = 0.5

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
        opt.selected_fold = fold_id
        prepare_dataset(opt)
        configure_logging(f"{opt.experiment_dir}/logs/train.log")

        #show loggings
        with open(os.path.join(opt.saver.experiment_dir, 'args.txt'), 'w') as args_file:
            for argument, value in sorted(vars(opt).items()):
                if type(value) in (str, int, float, tuple, list, bool):
                    args_file.write('{}: {}\n'.format(argument, value))

        with LoggingBlock("Commandline Arguments", emph=True):
            for argument, value in sorted(vars(opt).items()):
                if type(value) in (str, int, float, tuple, list):
                    logging.info('{}: {}'.format(argument, value))


        with LoggingBlock("Commandline Summary", emph=True):
            logging.info("{}Generator      :{} {}{}".format(blue, clear, opt.generator, clear))
            logging.info("{}Iterations     :{} {}{}".format(blue, clear, opt.niter, clear))
            logging.info("{}Rec. Weight    :{} {}{}".format(blue, clear, opt.rec_weight, clear))

        netG = GeneratorHPVAEGAN(opt).to(opt.device)
        if opt.netG != '':
            if not os.path.isfile(opt.netG):
                raise RuntimeError("=> no <G> checkpoint found at '{}'".format(opt.netG))
            checkpoint = torch.load(opt.netG, map_location=opt.device)
            opt.scale_idx = checkpoint['scale']
            opt.resumed_idx = checkpoint['scale']
            opt.resume_dir = '/'.join(opt.netG.split('/')[:-1])
            for _ in range(opt.scale_idx):
                netG.init_next_stage()
            netG.load_state_dict(checkpoint['state_dict'])
            opt.Noise_Amps = torch.load(os.path.join(opt.resume_dir, 'Noise_Amps.pth'), map_location=opt.device)['data']
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
    main()



