import argparse
import warnings
warnings.filterwarnings('ignore')

from utils.save_functions import *
from utils.get_functions import get_save_path
from MIS2D_Experiment import (dataset_argument,
                              MedicalImageSegmentationExperiment,
                              MultiClassMedicalImageSegmentationExperiment)

def MIS2D_main(args):
    print("Hello! We start experiment for 2D Medical Image Segmentation!")

    args = dataset_argument(args)
    print("Training Arguments : {}".format(args))

    if args.train_data_type in ['Synapse']: experiment = MultiClassMedicalImageSegmentationExperiment(args)
    else: experiment = MedicalImageSegmentationExperiment(args)

    test_results = experiment.fit()
    model_dirs = get_save_path(args)

    print("Save {} Model Test Results...".format(args.model_name))
    if args.train_data_type in ['Synapse']: save_multi_class_metrics(args, test_results, model_dirs)
    else: save_metrics(args, test_results, model_dirs)
#
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Following are the arguments that can be passed form the terminal itself!')

    # Data parameter
    parser.add_argument('--data_path', type=str, default='/media/jhnam0514/68334fe0-2b83-45d6-98e3-76904bf08127/home/namjuhyeon/Desktop/LAB/AwesomeDeepLearning/dataset/IS2D_dataset/BioMedicalDataset')
    parser.add_argument('--train_data_type', type=str, required=False, choices=['PolypSegData', 'ISIC2018', 'COVID19', 'BUSI', 'Synapse'])
    parser.add_argument('--test_data_type', type=str, required=False, choices=['CVC-ClinicDB', 'Kvasir-SEG', 'CVC-300', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'ISIC2018', 'PH2', 'COVID19', 'COVID19_2', 'BUSI', 'STU', 'Synapse', 'AMOS2022', 'AMOS2022_MRI'])
    parser.add_argument('--model_name', type=str, required=True, choices=['TransGUNet'])
    parser.add_argument('--save_path', type=str, default='model_weights')
    parser.add_argument('--print_step', type=int, default=10)

    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--transformer_backbone', type=str, default=None, choices=['p2t_small'])
    parser.add_argument('--pretrained', default=True, action='store_true')

    parser.add_argument('--our_method_configuration', type=str, required=False, default='baseline')
    parser.add_argument('--skip_channels', type=int, default=64)
    parser.add_argument('--top_k_channels', type=int, default=64)
    parser.add_argument('--frequency_selection', type=str, default='top')
    parser.add_argument('--num_scale_branch', type=int, default=3)
    parser.add_argument('--num_graph_layers', type=int, required=False, default=1)
    parser.add_argument('--num_heads', type=int, required=False, default=8)
    parser.add_argument('--boundary_threshold', type=float, default=0.4)

    args = parser.parse_args()

    if args.train_data_type == 'ISIC2018': args.test_data_type = 'ISIC2018'
    if args.train_data_type == 'COVID19': args.test_data_type = 'COVID19'
    if args.train_data_type == 'BUSI': args.test_data_type = 'BUSI'
    if args.train_data_type == 'PolypSegData': args.test_data_type = 'CVC-ClinicDB'
    if args.train_data_type == 'Synapse': args.test_data_type = 'Synapse'

    MIS2D_main(args)

    # Skin Cancer Segmentation Generalizability Test
    if args.train_data_type == 'ISIC2018':
        for test_data_type in ['ISIC2018', 'PH2']:
            args.train = False
            args.test_data_type = test_data_type

            if test_data_type == 'ISIC2018': continue

            MIS2D_main(args)

    # COVID19 lesion Segmentation Generalizability Test
    if args.train_data_type == 'COVID19':
        for test_data_type in ['COVID19', 'COVID19_2']:
            args.train = False
            args.test_data_type = test_data_type

            if test_data_type == 'COVID19': continue

            MIS2D_main(args)

    # Polyp Segmentation Generalizability Test
    if args.train_data_type == 'PolypSegData':
        for test_data_type in ['Kvasir', 'CVC-300', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
        # for test_data_type in ['Kvasir']:
            if test_data_type == 'CVC-ClinicDB': continue
            args.train = False
            args.test_data_type = test_data_type

            MIS2D_main(args)

    # Ultrasound Tumor Segmentation Generalizability Test
    if args.train_data_type == 'BUSI':
        for test_data_type in ['BUSI', 'STU']:
            if test_data_type == 'BUSI': continue
            args.train = False
            args.test_data_type = test_data_type

            MIS2D_main(args)

    if args.train_data_type == 'Synapse':
        for test_data_type in ['Synapse', 'AMOS2022', 'AMOS2022_MRI']:
            args.train = False
            args.test_data_type = test_data_type

            if test_data_type == 'Synapse': continue

            MIS2D_main(args)