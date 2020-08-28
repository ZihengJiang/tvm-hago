import tvm
from tvm import relay
from mxnet import gluon

from common.dataset_prep import MXNetImagenetDatasetPreparator as DatasetPreparator
from common.dataset_prep import prep_calibration_iter
from common.model_compiler import compile_and_run
from common.quantize_helper import quantize
import numpy as np
import argparse


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
def get_model():
    gluon_model = gluon.model_zoo.vision.get_model(model_name, pretrained=True)
    img_size = 299 if model_name == 'inceptionv3' else 224
    data_shape = (batch_size, 3, img_size, img_size)
    mod, params = relay.frontend.from_mxnet(gluon_model, {"data": data_shape})
    return mod, params


def main():
    val_path = '/home/ubuntu/tensorflow_datasets/downloads/manual/imagenet2012/val'
    num_calib_samples = 100
    num_test_samples = args.val_samples
    dataset_preparator = DatasetPreparator(val_path, num_calib_samples, num_test_samples, batch_size)
    img_size = 299 if model_name == 'inceptionv3' else 224
    print(img_size)
    
    val_dataset = dataset_preparator.preprocess_val(img_size, 'float32')

    # Original 
    fp32_mod, params = get_model()
    compile_and_run(fp32_mod, params, target, "mxnet_" + model_name + "_fp32",
                    val_dataset, 'data', ctx)

    
    # Data aware 
    calib_dataset = dataset_preparator.preprocess_calib(img_size, 'float32')
    c = prep_calibration_iter(calib_dataset, 'data')
    fp32_mod, params = get_model()
    mod = quantize(fp32_mod, params, True, c)
    compile_and_run(mod, params, target, "mxnet_" + model_name + "_calibration",
                    val_dataset, 'data', ctx)


if __name__ == '__main__':
    main()
