import torch
from model.model import *
from external.e2vid_plus.model.model import FlowNet

import os
import sys
# insert the `external/e2vid_plus` directory into the module search path
PROJECT_DIR = os.path.join(os.path.dirname(__file__), "..", "external", "e2vid_plus")
sys.path.insert(1, PROJECT_DIR)



def load_model(path_to_model):
    print('Loading model {}...'.format(path_to_model))
    raw_model = torch.load(path_to_model)
    arch = raw_model['arch']

    if arch == "FlowNet":   # E2VID+
        model_type = raw_model['config']['arch']['args']['unet_kwargs']
    else:   # E2VID
        try:
            model_type = raw_model['model']
        except KeyError:
            model_type = raw_model['config']['model']

    # instantiate model
    model = eval(arch)(model_type)

    # load model weights
    model.load_state_dict(raw_model['state_dict'])

    return model


def get_device(use_gpu):
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print('Device:', device)

    return device
