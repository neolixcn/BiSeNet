import argparse
import os.path as osp
import sys
sys.path.insert(0, '.')

import torch
import numpy as np
from lib.models import model_factory
from configs import cfg_factory
import onnxruntime


torch.set_grad_enabled(False)


parse = argparse.ArgumentParser()
parse.add_argument('--weight-path', dest='weight_pth', type=str,
        default='model_final.pth')
parse.add_argument('--outpath', dest='out_pth', type=str,
        default='model.onnx')
parse.add_argument('--cfg-file', dest='cfg_file', type=str, default='bisenetv2', help="specify the name without suffix of config file",)
parse.add_argument('--check-output', action='store_true', help="whether check the onnx output.")
args = parse.parse_args()


cfg = cfg_factory[args.cfg_file]
if cfg.use_sync_bn: cfg.use_sync_bn = False

# net = model_factory[cfg.model_type](19, output_aux=False)
net = model_factory[cfg.model_type](cfg.class_num, output_aux=False)
net.load_state_dict(torch.load(args.weight_pth), strict=False)
net.eval()

batch_size = 1
dummy_input = torch.randn(batch_size, 3, *cfg.cropsize)
input_names = ['input']
output_names = ['output']
dyn_axes = {'input':{0:'batch_size'}, 'output':{0:'batch_size'}}

torch.onnx.export(net,
                                dummy_input,
                                args.out_pth,
                                export_params=True,
                                opset_version=11,
                                verbose=True,
                                do_constant_folding=True,
                                input_names=input_names,
                                output_names=output_names,
                                dynamic_axes=dyn_axes,
                                )

# checkoutput
if args.check_output:
        torch_out = net(dummy_input)

        def to_numpy(tensor):
                return tensor.detach().cpu().numpy() if tensor.requires_grad == True else tensor.cpu().numpy()
        ort_session = onnxruntime.InferenceSession(args.out_pth)
        ort_inputs = {ort_session.get_inputs()[0].name : to_numpy(dummy_input)}
        ort_outs = ort_session.run(None, ort_inputs)

        print(np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-3, atol=1e-5))
        print("Export model has been tested with ONNXRuntime, and the result looks good!")