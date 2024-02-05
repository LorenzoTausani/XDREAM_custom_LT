from .deepsim import *
from .alexnet import *

#The load_net function takes a network name as an argument and returns an instance of the corresponding class from the get_net dictionary.

get_net = {
    'deepsim-norm1': DeePSiMNorm,
    'deepsim-norm2': DeePSiMNorm2,
    'deepsim-conv3': DeePSiMConv34,
    'deepsim-conv4': DeePSiMConv34,
    'deepsim-pool5': DeePSiMPool5,
    'deepsim-fc6': DeePSiMFc,
    'deepsim-fc7': DeePSiMFc,
    'deepsim-fc8': DeePSiMFc8,
    'alexnet': AlexNet,
}


def load_net(net_name):
    return get_net[net_name]()
