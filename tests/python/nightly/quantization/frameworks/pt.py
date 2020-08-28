import tvm
from tvm import relay

from common.dataset_prep import PytorchImagenetDatasetPreparator as DatasetPreparator
from common.dataset_prep import prep_calibration_iter
from common.model_compiler import compile_and_run
from common.quantize_helper import quantize
import numpy as np
import argparse

import torch
from torch.nn import Module
import torchvision


parser = argparse.ArgumentParser()
parser.add_argument("--val_samples", default=50000-40, type=int, help="number of validation samples")
parser.add_argument("--model", default="resnet50_v1", help="model to quantize")
args = parser.parse_args()

batch_size = 32
model_name = args.model
target = 'llvm -mcpu=skylake-avx512'
ctx = tvm.context(target)

###############################################################################
# Import the model
# ----------------
# We use the Relay MxNet frontend to import a model from the Gluon model zoo.
def load_model(model_name):
    """Given a model name, returns a model as well as an example input."""
    if hasattr(torchvision.models, model_name):
        with torch.no_grad():
            if model_name.startswith("inception"):
                height = width = 299
                mean = [0.5, 0.5, 0.5]
                std = [0.5, 0.5, 0.5]
            else:
                height = width = 224
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]
            input_shape = [batch_size, 3, height, width]
            input_data = torch.randn(input_shape).float()
            for channel in range(3):
                input_data[:, channel] -= mean[channel]
                input_data[:, channel] /= std[channel]
            model = getattr(torchvision.models, model_name)(pretrained=True)
            model = model.float().eval()
            return model, [input_data]
    try:
        import pretrainedmodels
        if hasattr(pretrainedmodels, model_name):
            return load_pretrainedmodels(model_name)
    except ModuleNotFoundError:
        raise ModuleNotFoundError("Please install pretrainedmodels.pytorch")
    raise RuntimeError("Model not supported")

def get_model():
    torch.set_grad_enabled(False)
    baseline_model, baseline_input = load_model(model_name)

    trace = torch.jit.trace(baseline_model, baseline_input)
    if isinstance(baseline_model, torch.nn.Module):
        trace = trace.float().eval()
        trace = trace.cpu()

    global input_names
    input_names = ["input{}".format(idx) for idx, inp in enumerate(baseline_input)]
    input_shapes = list(zip(input_names,
                            [inp.shape for inp in baseline_input]))
    mod, params = relay.frontend.from_pytorch(trace, input_shapes)
    return mod, params


def main():
    val_path = '/home/ubuntu/tensorflow_datasets/downloads/manual/imagenet2012/val'
    num_calib_samples = 100
    num_test_samples = args.val_samples
    dataset_preparator = DatasetPreparator(val_path, num_calib_samples, num_test_samples, batch_size)
    height = 224
    if model_name.startswith("inception"):
        height = 299

    
    val_dataset = dataset_preparator.preprocess_val(height, 'float32')

    # Original 
    fp32_mod, params = get_model()
    compile_and_run(fp32_mod, params, target, "pytorch_" + model_name + "_fp32",
                    val_dataset, input_names[0], ctx)
    
    # Data aware 
    fp32_mod, params = get_model()
    calib_dataset = dataset_preparator.preprocess_calib(height, 'float32')
    c = prep_calibration_iter(calib_dataset, input_names[0])
    mod = quantize(fp32_mod, params, True, c)
    compile_and_run(mod, params, target, "pytorch_" + model_name + "_calibration",
                    val_dataset, input_names[0], ctx)


if __name__ == '__main__':
    main()
