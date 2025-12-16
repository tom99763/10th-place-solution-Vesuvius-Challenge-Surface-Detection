import numpy as np
import torch
from copy import deepcopy
from batchgenerators.utilities.file_and_folder_operations import load_json, save_pickle
from nnunetv2.configuration import default_num_processes
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape

def export_prediction_from_logits(predicted_array_or_file,
                                  properties_dict,
                                  configuration_manager,
                                  plans_manager,
                                  dataset_json_dict_or_file,
                                  output_file_truncated,
                                  save_probabilities=False,
                                  num_threads_torch=8):

    if isinstance(dataset_json_dict_or_file, str):
        dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

    label_manager = plans_manager.get_label_manager(dataset_json_dict_or_file)

    ret = convert_predicted_logits_to_segmentation_with_correct_shape(
        predicted_array_or_file,
        plans_manager,
        configuration_manager,
        label_manager,
        properties_dict,
        return_probabilities=save_probabilities,
        num_threads_torch=num_threads_torch
    )
    return ret