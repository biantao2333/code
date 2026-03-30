import torch
import importlib

import sys
sys.path.append("../models")

net=importlib.import_module('models.HSCNN_plus')
hscnn_plus=net.hscnn_plus
net=importlib.import_module('models.AWAN')
AWAN=net.AWAN
net=importlib.import_module('models.MST_plus_plus')
MST_plus_plus=net.MST_Plus_Plus
net=importlib.import_module('models.HPRN')
HPRN=net.HPRN
net=importlib.import_module('models.GMSR')
GMSR=net.GMSR
net=importlib.import_module('models.mymamba')
MambaSSR=net.MambaSSR

def model_generator(method, pretrained_model_path=None):
    if method == 'HSCNN+':
        model = hscnn_plus(38,in_channels=3, out_channels=31)
    elif method == 'AWAN':
        model = AWAN(inplanes=3, planes=31,reduction=8)
    elif method == 'MST++':
        model = MST_plus_plus(in_channels=3, out_channels=31, n_feat=31, stage=3)
    elif method == 'HPRN':
        model = HPRN(inplanes=3, outplanes=31, interplanes=200, n_DRBs=10, window_size=8*8, n_scales=4, patch_num=4)
    elif method == 'GMSR':
        model = GMSR(inp_channels=3, out_channels=31)
    elif method =='MambaSSR':
        model=MambaSSR(in_channels=3, out_channels=31, dim=64, num_blocks=6)
    else:
        print(f'Method {method} is not defined !!!!')
    if pretrained_model_path is not None:
        print(f'load model from {pretrained_model_path}')
        checkpoint = torch.load(pretrained_model_path)
        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()},
                              strict=True)
    return model
