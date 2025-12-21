import pandas as pd
import json
from tqdm import tqdm
import numpy as np
import gc
import tifffile
import zipfile
import subprocess
import sys
import argparse
sys.path.append('/kaggle/working/nnunet')
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from queue import Queue
from threading import Thread
from typing import Tuple, Union, List
from queue import Queue, Empty
from omegaconf import DictConfig, OmegaConf
from threading import Thread
import itertools
import time
import numpy as np
import torch
sys.path.append('./src/nnUNet')
sys.path.append('./src/nnUNet/nnunetv2')
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from batchgenerators.utilities.file_and_folder_operations import load_json, join, isfile, subdirs
from batchgenerators.dataloading.data_loader import DataLoader
from torch._dynamo import OptimizedModule
from tqdm import tqdm
import copy
import traceback
from scipy.ndimage import gaussian_filter
import nnunetv2
from nnunetv2.configuration import default_num_processes
from nnunetv2.inference.sliding_window_prediction import compute_steps_for_sliding_window
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from batchgenerators.utilities.file_and_folder_operations import load_json, join, isfile, maybe_mkdir_p, isdir, subdirs, \
    save_json
from nnunetv2.utilities.file_path_utilities import get_output_folder, check_workers_alive_and_busy
from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape #,export_prediction_from_logits
from nnunetv2.utilities.json_export import recursive_fix_for_json_export
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from nnunetv2.inference.sliding_window_prediction import compute_gaussian, \
    compute_steps_for_sliding_window
from batchgenerators.utilities.file_and_folder_operations import load_json, save_pickle
from nnunetv2.configuration import default_num_processes
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from collections import OrderedDict
import inspect
from copy import deepcopy
import multiprocessing
from monai import transforms
import hydra
from hydra import initialize, compose
from omegaconf import OmegaConf
from PIL import Image, ImageSequence
import yaml
from pathlib import Path
from src.nnUNet_utils.export_custom import *
from monai.inferers.inferer import SlidingWindowInfererAdapt
from scipy.ndimage import (
    distance_transform_edt,
    maximum_filter,
    binary_fill_holes,
    label,
    find_objects,
)
from skimage.morphology import remove_small_objects, closing, ball
from skimage.segmentation import watershed
from scipy.ndimage import gaussian_filter
import cc3d
from src.models.deformNet3d import *
import shutil
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

#file paths
SPACING = [1, 1, 1]  # change if needed
CSV_PATH = './data/vesuvius-challenge-surface-detection/train.csv'
IMG_DIR = Path('./data/vesuvius-challenge-surface-detection/train_images')
BASE = Path("./nnUNet_raw_data_base/nnUNet_raw")
OUT_PATH = './submission.zip'
task_id = 900
task_name = "VesuviusScroll"
task_folder = f"Dataset{task_id:03d}_{task_name}"

base_dir = BASE / task_folder
imagesTr = base_dir / "imagesTs"
results = base_dir / "results"
selected_fold = 0 #modify this to run different fold

# make folders
imagesTr.mkdir(parents=True, exist_ok=True)
results.mkdir(parents=True,exist_ok=True)

#os.environ["nnUNet_raw"] = str(BASE) #"/kaggle/working/nnUNet_raw_data_base/nnUNet_raw"
os.environ["nnUNet_preprocessed"] = "./ft_nnunet/nnUNet/nnunet/preprocessed"
os.environ["nnUNet_results"] = "./nnunet"
os.environ['nnUNet_compile'] = 'False'
os.environ['torch.backends.cudnn.benchmark'] = 'True'

#modify this to run different fold
deformnet_ckpt_paths = [
    #'./vesuvius-codebase-v2/models/deform-dynunet-larger-patch-fold0-epoch=69-val_comp_metric=0.6536.ckpt',
    #'./vesuvius-codebase-v2/models/deform-dynunet-larger-patch-fold1-epoch=94-val_comp_metric=0.6511.ckpt',
    #'.vesuvius-codebase-v2/models/deform-dynunet-larger-patch-fold2-epoch=84-val_comp_metric=0.6536.ckpt',
    #'./vesuvius-codebase-v2/models/deform-dynunet-larger-patch-fold3-epoch=94-val_comp_metric=0.6535.ckpt',
    #'./vesuvius-codebase-v2/models/deform-dynunet-larger-patch-fold4-epoch=74-val_comp_metric=0.6475.ckpt',
    './vesuvius-codebase/models/deform-dynunet-fold0-epoch=114-val_comp_metric=0.6404.ckpt',
    #'.vesuvius-codebase/models/deform-dynunet-fold1-epoch=114-val_comp_metric=0.6336.ckpt',
    #'./vesuvius-codebase/models/deform-dynunet-fold2-epoch=139-val_comp_metric=0.6399.ckpt',
    #'./vesuvius-codebase/models/deform-dynunet-fold3-epoch=74-val_comp_metric=0.6398.ckpt',
    #'./vesuvius-codebase/models/deform-dynunet-fold4-epoch=99-val_comp_metric=0.6394.ckpt'
]
overlap = 0.5


def _getDefaultValue(env: str, dtype: type, default: any,) -> any:
    try:
        val = dtype(os.environ.get(env) or default)
    except:
        val = default
    return val

def get_parser():
    parser = argparse.ArgumentParser(description='Use this to run inference with nnU-Net. This function is used when '
                                                 'you want to manually specify a folder containing a trained nnU-Net '
                                                 'model. This is useful when the nnunet environment variables '
                                                 '(nnUNet_results) are not set.')
    parser.add_argument('-i', type=str, required=True,
                        help='input folder. Remember to use the correct channel numberings for your files (_0000 etc). '
                             'File endings must be the same as the training dataset!')
    parser.add_argument('-o', type=str, required=True,
                        help='Output folder. If it does not exist it will be created. Predicted segmentations will '
                             'have the same name as their source images.')
    parser.add_argument('-m', type=str, required=True,
                        help='Folder in which the trained model is. Must have subfolders fold_X for the different '
                             'folds you trained')
    parser.add_argument('-f', nargs='+', type=str, required=False, default=(0, 1, 2, 3, 4),
                        help='Specify the folds of the trained model that should be used for prediction. '
                             'Default: (0, 1, 2, 3, 4)')
    parser.add_argument('-step_size', type=float, required=False, default=0.5,
                        help='Step size for sliding window prediction. The larger it is the faster but less accurate '
                             'the prediction. Default: 0.5. Cannot be larger than 1. We recommend the default.')
    parser.add_argument('--disable_tta', action='store_true', required=False, default=False,
                        help='Set this flag to disable test time data augmentation in the form of mirroring. Faster, '
                             'but less accurate inference. Not recommended.')
    parser.add_argument('--verbose', action='store_true', help="Set this if you like being talked to. You will have "
                                                               "to be a good listener/reader.")
    parser.add_argument('--save_probabilities', action='store_true',
                        help='Set this to export predicted class "probabilities". Required if you want to ensemble '
                             'multiple configurations.')
    parser.add_argument('--continue_prediction', '--c', action='store_true',
                        help='Continue an aborted previous prediction (will not overwrite existing files)')
    parser.add_argument('-chk', type=str, required=False, default='checkpoint_final.pth',
                        help='Name of the checkpoint you want to use. Default: checkpoint_final.pth')
    parser.add_argument('-npp', type=int, required=False, default=3,
                        help='Number of processes used for preprocessing. More is not always better. Beware of '
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-nps', type=int, required=False, default=3,
                        help='Number of processes used for segmentation export. More is not always better. Beware of '
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-prev_stage_predictions', type=str, required=False, default=None,
                        help='Folder containing the predictions of the previous stage. Required for cascaded models.')
    parser.add_argument('-device', type=str, default='cuda', required=False,
                        help="Use this to set the device the inference should run with. Available options are 'cuda' "
                             "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use this to set which GPU ID! "
                             "Use CUDA_VISIBLE_DEVICES=X nnUNetv2_predict [...] instead!")
    parser.add_argument('--disable_progress_bar', action='store_true', required=False, default=False,
                        help='Set this flag to disable progress bar. Recommended for HPC environments (non interactive '
                             'jobs)')
    return parser.parse_args()


def load_model_from_checkpoint(model, ckpt_path):
    ckpt = torch.load(ckpt_path, weights_only=False)
    state_dict = ckpt['state_dict']
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        new_key = k.replace("model.", "") if k.startswith("model.") else k
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict)


def load_volume(path: Path) -> np.ndarray:
    """
    Load a multi-page TIFF into a 3D NumPy array: (slices, H, W)
    """
    try:
        with Image.open(path) as img:
            frames = [np.array(frame) for frame in ImageSequence.Iterator(img)]
        volume = np.stack(frames)
        return volume
    except Exception as e:
        raise RuntimeError(f"Error loading TIFF {path}: {e}")


def generate_transforms(
        transforms_config: list[dict],
) -> list[transforms.Transform]:
    transform_list = []
    # logger.debug(f"Generating {len(transforms_config)} transforms")

    for transform_config in transforms_config:
        transform_name = next(iter(transform_config))
        transform_kwargs = transform_config[transform_name]
        # logger.debug(
        #     f"Generating transform {transform_name} with kwargs {transform_kwargs}"
        # )
        transform: transforms.Transform = getattr(transforms, transform_name)(
            **transform_kwargs
        )  # type: ignore
        transform_list.append(transform)
    return transforms.Compose(transform_list)


def gaussian_kernel_3d(kernel_size=5, sigma=1.0, device="cuda"):
    """Returns a normalized 3D Gaussian kernel (1,1,K,K,K)."""
    ax = torch.arange(kernel_size, device=device) - kernel_size // 2
    xx, yy, zz = torch.meshgrid(ax, ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2 + zz**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel


def gaussian_blur_3d(x, kernel_size=7, sigma=10.0):
    """
    x: (B, C, D, H, W)
    """
    B, C, D, H, W = x.shape
    kernel = gaussian_kernel_3d(kernel_size, sigma, device=x.device)

    # shape: (C, 1, K, K, K)
    kernel = kernel.expand(C, 1, kernel_size, kernel_size, kernel_size)

    # depthwise convolution
    return F.conv3d(x, kernel, padding=kernel_size // 2, groups=C)


class DeformNNUnetPredictor3D(nnUNetPredictor):
    def __init__(self, deformnet_cfg=None, **kwargs):
        super().__init__(**kwargs)
        self.deformnet_cfg = deformnet_cfg

    def init_deformnet_and_transforms(self):
        self.deformnet_list = []
        for ckpt_path in deformnet_ckpt_paths:
            m = DeformDynUnet(self.deformnet_cfg).to(self.device)
            m.eval()
            load_model_from_checkpoint(m, ckpt_path)
            self.deformnet_list.append(m)
        self.deformnet_transforms = generate_transforms(self.deformnet_cfg.data.transforms.test)
        self.sliding_window_inferer = SlidingWindowInfererAdapt(
            roi_size=self.deformnet_cfg.input_size, sw_batch_size=1, overlap=overlap, mode="constant",
            progress=True
        )

    @torch.inference_mode()
    def predict_and_export_with_deformnet(self,
                                          ret,
                                          output_file_truncated,
                                          properties_dict,
                                          plans_manager,
                                          dataset_json_dict_or_file
                                          ):
        prob_mask = ret.get()[0][1][1]  # (d, h, w)
        case_id = output_file_truncated.split('/')[-1]
        _data_path = IMG_DIR / f'{case_id}.tif'
        vol = load_volume(_data_path)
        raw = {"Image": vol, "Mask_OOF": prob_mask}
        _data = self.deformnet_transforms(raw)
        vol, prob_mask = _data['Image'][None,], _data['Mask_OOF'][None,]
        vol = vol.to(self.device)
        prob_mask = prob_mask.to(self.device)
        prev_mask_pred = (prob_mask > self.deformnet_cfg.threshold).float()
        prev_mask_pred = gaussian_blur_3d(prev_mask_pred)
        x = torch.cat([vol, prev_mask_pred], dim=1)
        predictions = []
        for m in self.deformnet_list:
            pred_warped = self.sliding_window_inferer(x, m)
            predictions.append(pred_warped)
        prediction_avg = torch.cat(predictions, dim=0).mean(dim=0)  # (b, c, d, h, w)
        segmentation_final = prediction_avg > self.deformnet_cfg.threshold
        segmentation_final = segmentation_final[0].cpu().numpy()  # .astype('uint8')
        segmentation_prob = prediction_avg[0].cpu().numpy()
        # segmentation_final = postprocess_mask(segmentation_final)
        del vol, prob_mask, pred_warped

        # export
        if isinstance(dataset_json_dict_or_file, str):
            dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)
        rw = plans_manager.image_reader_writer_class()
        rw.write_seg(segmentation_final,
                     output_file_truncated + dataset_json_dict_or_file['file_ending'],
                     properties_dict)
        np.savez(
            output_file_truncated + '.npz',
            prob=segmentation_prob
        )
        del segmentation_final, segmentation_prob
        torch.cuda.empty_cache()

    def predict_from_data_iterator(self,
                                   data_iterator,
                                   save_probabilities: bool = False,
                                   num_processes_segmentation_export: int = default_num_processes):
        """
        each element returned by data_iterator must be a dict with 'data', 'ofile' and 'data_properties' keys!
        If 'ofile' is None, the result will be returned instead of written to a file
        """
        with multiprocessing.get_context("spawn").Pool(num_processes_segmentation_export) as export_pool:
            worker_list = [i for i in export_pool._pool]
            r = []
            for preprocessed in data_iterator:
                data = preprocessed['data']
                if isinstance(data, str):
                    delfile = data
                    data = torch.from_numpy(np.load(data))
                    os.remove(delfile)

                ofile = preprocessed['ofile']
                print('**ofile**: ', ofile)
                if ofile is not None:
                    print(f'\nPredicting {os.path.basename(ofile)}:')
                else:
                    print(f'\nPredicting image of shape {data.shape}:')

                print(f'perform_everything_on_device: {self.perform_everything_on_device}')

                properties = preprocessed['data_properties']

                # let's not get into a runaway situation where the GPU predicts so fast that the disk has to be swamped with
                # npy files
                proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)
                while not proceed:
                    time.sleep(0.1)
                    proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)

                # convert to numpy to prevent uncatchable memory alignment errors from multiprocessing serialization of torch tensors
                prediction = self.predict_logits_from_preprocessed_data(data).cpu().detach().numpy()

                ret = export_pool.starmap_async(
                    export_prediction_from_logits,
                    ((prediction, properties, self.configuration_manager, self.plans_manager,
                      self.dataset_json, ofile, save_probabilities),)
                )

                self.predict_and_export_with_deformnet(ret, ofile, properties,
                                                       self.plans_manager, self.dataset_json
                                                       )

                if ofile is not None:
                    print(f'done with {os.path.basename(ofile)}')
                else:
                    print(f'\nDone with image of shape {data.shape}:')
            # _ret = [i.get()[0] for i in r]

        if isinstance(data_iterator, MultiThreadedAugmenter):
            data_iterator._finish()

        # clear lru cache
        compute_gaussian.cache_clear()
        # clear device cache
        empty_cache(self.device)

    def predict_from_files(self,
                           list_of_lists_or_source_folder: Union[str, List[List[str]]],
                           output_folder_or_list_of_truncated_output_files: Union[str, None, List[str]],
                           save_probabilities: bool = False,
                           overwrite: bool = True,
                           num_processes_preprocessing: int = default_num_processes,
                           num_processes_segmentation_export: int = default_num_processes,
                           folder_with_segs_from_prev_stage: str = None,
                           num_parts: int = 1,
                           part_id: int = 0):
        """
        This is nnU-Net's default function for making predictions. It works best for batch predictions
        (predicting many images at once).
        """
        assert part_id <= num_parts, ("Part ID must be smaller than num_parts. Remember that we start counting with 0. "
                                      "So if there are 3 parts then valid part IDs are 0, 1, 2")
        if isinstance(output_folder_or_list_of_truncated_output_files, str):
            output_folder = output_folder_or_list_of_truncated_output_files
        elif isinstance(output_folder_or_list_of_truncated_output_files, list):
            output_folder = os.path.dirname(output_folder_or_list_of_truncated_output_files[0])
        else:
            output_folder = None

        ########################
        # let's store the input arguments so that its clear what was used to generate the prediction
        if output_folder is not None:
            my_init_kwargs = {}
            for k in inspect.signature(self.predict_from_files).parameters.keys():
                my_init_kwargs[k] = locals()[k]
            my_init_kwargs = deepcopy(
                my_init_kwargs)  # let's not unintentionally change anything in-place. Take this as a
            recursive_fix_for_json_export(my_init_kwargs)
            maybe_mkdir_p(output_folder)
            save_json(my_init_kwargs, join(output_folder, 'predict_from_raw_data_args.json'))

            # we need these two if we want to do things with the predictions like for example apply postprocessing
            save_json(self.dataset_json, join(output_folder, 'dataset.json'), sort_keys=False)
            save_json(self.plans_manager.plans, join(output_folder, 'plans.json'), sort_keys=False)
        #######################

        # check if we need a prediction from the previous stage
        if self.configuration_manager.previous_stage_name is not None:
            assert folder_with_segs_from_prev_stage is not None, \
                f'The requested configuration is a cascaded network. It requires the segmentations of the previous ' \
                f'stage ({self.configuration_manager.previous_stage_name}) as input. Please provide the folder where' \
                f' they are located via folder_with_segs_from_prev_stage'

        # sort out input and output filenames
        list_of_lists_or_source_folder, output_filename_truncated, seg_from_prev_stage_files = \
            self._manage_input_and_output_lists(list_of_lists_or_source_folder,
                                                output_folder_or_list_of_truncated_output_files,
                                                folder_with_segs_from_prev_stage, overwrite, part_id, num_parts,
                                                save_probabilities)
        if len(list_of_lists_or_source_folder) == 0:
            return

        data_iterator = self._internal_get_data_iterator_from_lists_of_filenames(list_of_lists_or_source_folder,
                                                                                 seg_from_prev_stage_files,
                                                                                 output_filename_truncated,
                                                                                 num_processes_preprocessing)

        self.predict_from_data_iterator(data_iterator, save_probabilities, num_processes_segmentation_export)



def predict(cfg):
    args = get_parser()
    args.f = [i if i == 'all' else int(i) for i in args.f]
    #model_folder = get_output_folder(args.d, args.tr, args.p, args.c)

    if not isdir(args.o):
        maybe_mkdir_p(args.o)

    assert args.device in ['cpu', 'cuda',
                           'mps'], f'-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {args.device}.'
    if args.device == 'cpu':
        # let's allow torch to use hella threads
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    elif args.device == 'cuda':
        # multithreading in torch doesn't help nnU-Net if run on GPU
        #torch.set_num_threads(1)
        #torch.set_num_interop_threads(1)
        device = torch.device('cuda')
    else:
        device = torch.device('mps')

    predictor = DeformNNUnetPredictor3D(tile_step_size=args.step_size,
                                        use_gaussian=True,
                                        use_mirroring=not args.disable_tta,
                                        perform_everything_on_device=True,
                                        device=device,
                                        verbose=args.verbose,
                                        verbose_preprocessing=args.verbose,
                                        allow_tqdm=not args.disable_progress_bar,
                                        deformnet_cfg = cfg
                                       )
    predictor.initialize_from_trained_model_folder(
        args.m,
        args.f,
        checkpoint_name=args.chk
    )
    predictor.init_deformnet_and_transforms()
    predictor.predict_from_files(args.i, args.o, save_probabilities=args.save_probabilities,
                                    overwrite=not args.continue_prediction,
                                    num_processes_preprocessing=args.npp,
                                    num_processes_segmentation_export=args.nps,
                                    folder_with_segs_from_prev_stage=args.prev_stage_predictions,
                                    num_parts=1,
                                    part_id=0
                                )

def main():
    json_path = './splits_final.json'
    with open(json_path, 'r') as f:
        val_splits = json.load(f)
    val_ids = val_splits[selected_fold]['val']
    val_ids = list(map(lambda x: int(x), val_ids))
    df = pd.read_csv(CSV_PATH)
    df = df[df.id.isin(val_ids)]

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Copying TIFFs"):
        case_id = str(row["id"])
        scroll_id = str(row["scroll_id"])

        img_path = IMG_DIR / f"{case_id}.tif"
        # lbl_path = LBL_DIR / f"{case_id}.tif"

        if not img_path.exists():
            print(f"Skipping missing pair: {scroll_id}")
            continue

        # destination names for nnUNet
        img_dst = imagesTr / f"{case_id}_0000.tif"
        # lbl_dst = labelsTr / f"{case_id}.tif"

        json_dst_img = imagesTr / f"{case_id}.json"
        # json_dst_lbl = labelsTr / f"{case_id}.json"

        # ---- COPY FILES (FAST) ----
        shutil.copy2(img_path, img_dst)
        # shutil.copy2(lbl_path, lbl_dst)

        # ---- WRITE SPACING JSON ----
        spacing_info = {"spacing": SPACING}

        with open(json_dst_img, "w") as f:
            json.dump(spacing_info, f)

    sys.argv = [
        'predict.py',  # dummy script name
        '-i', './nnUNet_raw_data_base/nnUNet_raw/Dataset900_VesuviusScroll/imagesTs',
        '-o', './nnUNet_raw_data_base/nnUNet_raw/Dataset900_VesuviusScroll/results',
        '-m', 'nnunet/nnUNet_results/Dataset900_VesuviusScroll/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres',
        '--disable_tta',
        '--save_probabilities',
        '-f', '0', '1', '2', '3', '4',
    ]

    SRC = Path("./vesuvius-codebase/configs")
    DST = Path("./configs_")
    DST.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SRC, DST, dirs_exist_ok=True)

    with initialize(version_base=None, config_path="configs_"):
        cfg = compose(config_name="config_deform")

    predict(cfg)


if __name__ == "__main__":
    main()