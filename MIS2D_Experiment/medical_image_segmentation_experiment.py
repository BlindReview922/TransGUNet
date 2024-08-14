import torch

import torchvision.transforms as transforms

import numpy as np
from tqdm import tqdm

from utils.load_functions import load_model
from utils.calculate_metrics import MIS_Metrics_Calculator
from MIS2D_Experiment._MIS2Dbase import BaseMedicalImageSegmentationExperiment

class MedicalImageSegmentationExperiment(BaseMedicalImageSegmentationExperiment):
    def __init__(self, args):
        super(MedicalImageSegmentationExperiment, self).__init__(args)

        self.size_rates = [0.75, 1, 1.25] if self.args.multi_scale_train else [1]

    def fit(self):
        print("INFERENCE")
        self.model = load_model(self.args, self.model)
        test_results = self.inference()

        return test_results

    def inference(self):
        self.model.eval()

        self.metrics_calculator = MIS_Metrics_Calculator(self.args.metric_list)
        total_metrics_dict = self.metrics_calculator.total_metrics_dict

        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm(self.test_loader)):
                output_dict = self.forward(data)

                for idx, (target_, output_) in enumerate(zip(data['target'], output_dict['prediction'])):
                    predict = torch.sigmoid(output_).squeeze()
                    metrics_dict = self.metrics_calculator.get_metrics_dict(predict, target_)

                    for metric in self.metrics_calculator.metrics_list:
                        total_metrics_dict[metric].append(metrics_dict[metric])

        for metric in self.metrics_calculator.metrics_list:
            total_metrics_dict[metric] = np.round(np.mean(total_metrics_dict[metric]), 4)

        return total_metrics_dict


    def transform_generator(self):
        transform_list = [
            transforms.Resize((self.args.image_size, self.args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.args.mean, std=self.args.std),
        ]

        target_transform_list = [
            transforms.Resize((self.args.image_size, self.args.image_size)),
            transforms.ToTensor(),
        ]

        return transforms.Compose(transform_list), transforms.Compose(target_transform_list)