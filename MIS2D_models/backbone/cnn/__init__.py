import sys

import torch
import torch.utils.model_zoo as model_zoo

import MIS2D_models.backbone.cnn as cnn_backbone

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'res2net50_v1b_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_v1b_26w_4s-3cf99910.pth',
    'resnest50': 'https://github.com/zhanghang1989/ResNeSt/releases/download/weights_step1/528c19ca-resnest50.pth'
}

def load_cnn_backbone_model(backbone_name, pretrained=False, **kwargs):
    if backbone_name=='resnet18':
        from MIS2D_models.backbone.cnn.resnet import ResNet
        model = ResNet(cnn_backbone.resnet.BasicBlock, [2, 2, 2, 2], **kwargs)
    elif backbone_name=='resnet50':
        from MIS2D_models.backbone.cnn.resnet import ResNet
        model = ResNet(cnn_backbone.resnet.Bottleneck, [3, 4, 6, 3], **kwargs)
    elif backbone_name=='res2net50_v1b_26w_4s':
        from MIS2D_models.backbone.cnn.res2net import Res2Net
        model = Res2Net(cnn_backbone.res2net.Bottle2Neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
    elif backbone_name=='resnest50':
        from MIS2D_models.backbone.cnn.resnest import ResNeSt
        model = ResNeSt(cnn_backbone.resnest.Bottleneck, [3, 4, 6, 3],
                        radix=2, groups=1, bottleneck_width=64,
                        deep_stem=True, stem_width=32, avg_down=True,
                        avd=True, avd_first=False, **kwargs)
    else:
        print("Invalid backbone")
        sys.exit()

    if pretrained:
        if backbone_name == 'resnest50':
            _url_format = 'https://github.com/zhanghang1989/ResNeSt/releases/download/weights_step1/{}-{}.pth'

            _model_sha256 = {name: checksum for checksum, name in [
                ('528c19ca', 'resnest50'),
                ('22405ba7', 'resnest101'),
                ('75117900', 'resnest200'),
                ('0cc87c48', 'resnest269'),
            ]}

            def short_hash(name):
                if name not in _model_sha256:
                    raise ValueError('Pretrained model for {name} is not available.'.format(name=name))
                return _model_sha256[name][:8]

            resnest_model_urls = {name: _url_format.format(name, short_hash(name)) for
                                  name in _model_sha256.keys()
                                  }

            model.load_state_dict(torch.hub.load_state_dict_from_url(
                resnest_model_urls['resnest50'], progress=True, check_hash=True))
        else:
            model.load_state_dict(model_zoo.load_url(model_urls[backbone_name]))

        print("Complete loading your pretrained backbone {}".format(backbone_name))
    return model
