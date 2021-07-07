
from .bisenetv1 import cfg as bisenetv1_cfg
from .bisenetv2 import cfg as bisenetv2_cfg
from .bisenetv2_combined import cfg as bisenetv2_combined_cfg
from .bisenetv2_org import cfg as bisenetv2_org_cfg
from .bisenetv2_cityscapes import cfg as bisenetv2_cityscapes_cfg
from .bisenetv2_neolix_fisheye import cfg as bisenetv2_neolix_fisheye_cfg
from .bisenetv2_neolix_fisheye2 import cfg as bisenetv2_neolix_fisheye_cfg2
from .bisenetv2_neolix_test import cfg as bisenetv2_neolix_test_cfg



class cfg_dict(object):

    def __init__(self, d):
        self.__dict__ = d


cfg_factory = dict(
    bisenetv1=cfg_dict(bisenetv1_cfg),
    bisenetv2=cfg_dict(bisenetv2_cfg),
    bisenetv2_combined=cfg_dict(bisenetv2_combined_cfg),
    bisenetv2_org=cfg_dict(bisenetv2_org_cfg),
    bisenetv2_cityscapes = cfg_dict(bisenetv2_cityscapes_cfg),
    bisenetv2_neolix_fisheye = cfg_dict(bisenetv2_neolix_fisheye_cfg),
    bisenetv2_neolix_fisheye2 = cfg_dict(bisenetv2_neolix_fisheye_cfg2),
    bisenetv2_neolix_test = cfg_dict(bisenetv2_neolix_test_cfg),
)
