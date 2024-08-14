import sys

from torch.utils.data import DataLoader

from utils.get_functions import get_deivce
from MIS2D_models import MIS2D_model, model_to_device
from dataset.PolypSegData import PolypImageSegDataset
from dataset.SkinSegmentation2018Dataset import *
from dataset.PH2Dataset import *
from dataset.Covid19CTScanDataset import *
from dataset.Covid19CTScan2Dataset import *
from dataset.BUSISegmentationDataset import *
from dataset.STUSegmentationDataset import *
from dataset.SynapseDataset import *
from dataset.AMOS2022Dataset import *

class BaseMedicalImageSegmentationExperiment(object):
    def __init__(self, args):
        super(BaseMedicalImageSegmentationExperiment, self).__init__()

        self.args = args
        self.args.device = get_deivce()

        self.test_loader = self.dataloader_generator()

        print("STEP2. Load 2D Image Segmentation Model {}...".format(self.args.model_name))
        self.model = MIS2D_model(args)
        self.model = model_to_device(self.args, self.model)


    def forward(self, data):
        data = self.cpu_to_gpu(data)

        with torch.cuda.amp.autocast():
            output_dict = self.model(data)

        return output_dict

    def cpu_to_gpu(self, data):
        for key in ['image', 'target']:
            data[key] = data[key].to(self.args.device)

        return data

    def history_generator(self):
        self.history = dict()
        self.history['train_loss'] = list()
        self.history['val_loss'] = list()

    def worker_init_fn(self, worker_id):
        random.seed(4321 + worker_id)

    def dataloader_generator(self):
        test_image_transform, test_target_transform = self.transform_generator()

        print("STEP1. Load {} Test Dataset Loader...".format(self.args.test_data_type))
        if self.args.test_data_type in ['CVC-ClinicDB', 'Kvasir', 'CVC-300', 'CVC-ColonDB', 'ETIS-LaribPolypDB'] :
            test_dataset = PolypImageSegDataset(self.args, self.args.test_dataset_dir, mode='test', transform=test_image_transform, target_transform=test_target_transform)
        elif self.args.test_data_type == 'ISIC2018':
            test_dataset = ISIC2018Dataset(self.args, self.args.test_dataset_dir, mode='test', transform=test_image_transform, target_transform=test_target_transform)
        elif self.args.test_data_type == 'PH2':
            test_dataset = PH2Dataset(self.args, self.args.test_dataset_dir, mode='test', transform=test_image_transform, target_transform=test_target_transform)
        elif self.args.test_data_type == 'COVID19':
            test_dataset = Covid19CTScanDataset(self.args, self.args.test_dataset_dir, mode='test', transform=test_image_transform, target_transform=test_target_transform)
        elif self.args.test_data_type == 'COVID19_2':
            test_dataset = Covid19CTScan2Dataset(self.args, self.args.test_dataset_dir, mode='test', transform=test_image_transform, target_transform=test_target_transform)
        elif self.args.test_data_type in 'BUSI':
            test_dataset = BUSISegmentationDataset(self.args, self.args.test_dataset_dir, mode='test', transform=test_image_transform, target_transform=test_target_transform)
        elif self.args.test_data_type == 'STU':
            test_dataset = STUSegmentationDataset(self.args, self.args.test_dataset_dir, mode='test',  transform=test_image_transform, target_transform=test_target_transform)
        elif self.args.test_data_type == 'Synapse':
            test_dataset = SynapseDataset(self.args, self.args.test_dataset_dir, mode='test', transform=None, target_transform=test_target_transform)
        elif self.args.test_data_type in ['AMOS2022', 'AMOS2022_MRI']:
            test_dataset = AMOS2022Dataset(self.args, self.args.test_dataset_dir, mode='test',
                                           transform=None,
                                           target_transform=test_target_transform)
        else:
            print("Wrong Dataset")
            sys.exit()

        test_loader = DataLoader(test_dataset,
                                       batch_size=self.args.test_batch_size,
                                       shuffle=False,
                                       num_workers=self.args.num_workers,
                                       pin_memory=True)

        return test_loader