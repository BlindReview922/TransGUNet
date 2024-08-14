import os
import sys

from MIS2D_models import _training_configurations
from .medical_image_segmentation_experiment import MedicalImageSegmentationExperiment
from .multi_class_medical_image_segmentation_experiment import MultiClassMedicalImageSegmentationExperiment

def dataset_argument(args):
    try:
        args.train_dataset_dir = os.path.join(args.data_path, args.train_data_type)
        args.test_dataset_dir = os.path.join(args.data_path, args.test_data_type)
    except TypeError:
        print("join() argument must be str, bytes, or os.PathLike object, not 'NoneType'")
        print("Please explicitely write the dataset type")
        sys.exit()

    args = _training_configurations(args)

    return args