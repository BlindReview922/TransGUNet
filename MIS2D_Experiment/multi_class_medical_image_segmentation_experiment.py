from time import time

import torch

import torchvision.transforms as transforms

import numpy as np
from medpy import metric
from scipy.ndimage import zoom

from utils.load_functions import load_model
from utils.calculate_metrics import MCMIS_Metrics_Calculator
from MIS2D_Experiment._MIS2Dbase import BaseMedicalImageSegmentationExperiment

class MultiClassMedicalImageSegmentationExperiment(BaseMedicalImageSegmentationExperiment):
    def __init__(self, args):
        super(MultiClassMedicalImageSegmentationExperiment, self).__init__(args)
        self.args = args

        if self.args.num_classes == 9: class_name_list = ['spleen', 'right_kidney', 'left_kidney', 'gallbladder', 'pancreas', 'liver', 'stomach', 'aorta']
        else: class_name_list = ['spleen', 'right_kidney', 'left_kidney', 'gallbladder', 'esophagus', 'liver', 'stomach', 'aorta', 'inferior_vena_cava', 'portal_vein_and_splenic_vein', 'pancreas', 'right_adrenal_gland', 'left_adrenal_gland']

        self.args.class_name_list = class_name_list
        self.metrics_calculator = MCMIS_Metrics_Calculator(args.metric_list, class_name_list)

    def fit(self):
        print("INFERENCE")
        self.model = load_model(self.args, self.model)
        test_results = self.inference()

        return test_results

    def inference(self):
        metric_list = 0.0
        total_metrics_dict = self.metrics_calculator.total_metrics_dict

        for i_batch, sampled_batch in enumerate(self.test_loader):
            image, label, case_name = sampled_batch["image"], sampled_batch["target"], sampled_batch['case_name'][0]

            print("Case Number : ", i_batch + 1, " | Case Name : ", case_name, " | Shape : ", image.shape, " | Label Shape : ", label.shape)
            with torch.cuda.amp.autocast():
                metric_i = self.test_single_volume(image, label, i_batch, case_name)
            metric_list += np.array(metric_i)
            print('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))

        metric_list = metric_list / len(self.test_loader)

        for class_idx, class_name in enumerate(self.args.class_name_list):
            total_metrics_dict[class_name]['DSC'] = metric_list[class_idx][0]
            total_metrics_dict[class_name]['HD95'] = metric_list[class_idx][1]
            total_metrics_dict[class_name]['IoU'] = metric_list[class_idx][2]

        return total_metrics_dict

    def test_single_volume(self, image, label, i_batch, case_name):
        image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()

        if len(image.shape) == 3:
            prediction = np.zeros_like(label)

            for slice_idx in range(image.shape[0]):
                image_slice = image[slice_idx, :, :]
                label_slice = label[slice_idx, :, :]

                x, y = image_slice.shape[0], image_slice.shape[1]

                if x != self.args.image_size or y != self.args.image_size:
                    image_slice = zoom(image_slice, (self.args.image_size / x, self.args.image_size / y), order=3)
                    label_slice = zoom(label_slice, (self.args.image_size / x, self.args.image_size / y), order=0)

                image_slice = torch.from_numpy(image_slice).unsqueeze(0).unsqueeze(0).float().to(self.args.device)
                label_slice = torch.from_numpy(label_slice).unsqueeze(0).long().to(self.args.device)

                data = {'image': image_slice, 'target': label_slice}

                self.model.eval()

                with torch.no_grad():
                    output_dict = self.model(data)

                    prediction_slice = torch.argmax(torch.softmax(output_dict['prediction'], dim=1), dim=1).squeeze(0)
                    prediction_slice = prediction_slice.cpu().detach().numpy()

                    if x != self.args.image_size or y != self.args.image_size:
                        pred = zoom(prediction_slice, (x / self.args.image_size, y / self.args.image_size), order=0)
                    else:
                        pred = prediction_slice

                    prediction[slice_idx, :, :] = pred

        start_time = time()
        metric_list = []
        for i in range(1, self.args.num_classes):
            metric_list.append(calculate_metric_percase(prediction == i, label == i))
        print("Time : ", time() - start_time)

        return metric_list

    def transform_generator(self):
        transform_list = None
        target_transform_list = None

        return transforms.Compose(transform_list), transforms.Compose(target_transform_list)

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1; gt[gt > 0] = 1

    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.dc(pred, gt)
        hd95 = metric.hd95(pred, gt)
        iou = metric.jc(pred, gt)
        # asd = metric.asd(pred, gt)

        return dice, hd95, iou#, asd
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0, 1#, 1
    else:
        return 0, 0, 0#, 0