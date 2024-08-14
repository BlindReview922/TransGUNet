import torch
import torch.nn as nn
import torch.nn.functional as F

def MIS2D_model(args):
    from MIS2D_models.ours2 import Ours2
    return Ours2(args=args,
                 num_channels=args.num_channels,
                 num_classes=args.num_classes,
                 num_graph_layers=args.num_graph_layers,
                 skip_channels=args.skip_channels,
                 top_k_channels=args.top_k_channels,
                 transformer_backbone=args.transformer_backbone,
                 pretrained=args.pretrained,
                 num_heads=args.num_heads,
                 boundary_threshold=args.boundary_threshold)

def model_to_device(args, model):
    if torch.cuda.device_count() > 1:
        print('Multi GPU activate : {}'.format(torch.cuda.device_count()))
        model = nn.DataParallel(model).to(args.device)
    else:
        model = model.to(args.device)

    return model

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()

def _training_configurations(args):
    from MIS2D_models.ours2 import _training_config
    args = _training_config(args)

    return args