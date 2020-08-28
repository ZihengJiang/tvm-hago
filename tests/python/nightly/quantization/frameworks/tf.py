try:
  # %tensorflow_version only exists in Colab.
  import tensorflow.compat.v2 as tf
except Exception:
  pass
# tf.enable_v2_behavior()
import tensorflow_hub as hub

import tvm
from tvm import relay

from common.dataset_prep import TFImagenetDatasetPreparator as DatasetPreparator
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
target = 'llvm'
ctx = tvm.context(target)


##############################
# Original FP32 TF/Keras model
##############################
tf_hub_links = {
    "resnet50"             : "https://tfhub.dev/tensorflow/resnet_50/classification/1",
    "resnet_v2_50"          : "https://tfhub.dev/google/imagenet/resnet_v2_50/classification/4",
    "mobilenet_v1"          : "https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/classification/4",
    "mobilenet_v2"          : "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/4",
    "inception_v1"          : "https://tfhub.dev/google/imagenet/inception_v1/classification/4",
    "inception_v2"          : "https://tfhub.dev/google/imagenet/inception_v2/classification/4",
    "inception_v3"          : "https://tfhub.dev/google/imagenet/inception_v3/classification/4",
    "inception_v3_preview"  : "https://tfhub.dev/google/tf2-preview/inception_v3/classification/4",
    "mobilenet_v2_preview"  : "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4",
    # "efficientnet_b0"       : "https://tfhub.dev/tensorflow/efficientnet/b0/classification/1",
}


###############################################################################
# Import the model
# ----------------
# We use the Relay MxNet frontend to import a model from the Gluon model zoo.
def get_model():

    relay_file = "relay.json"
    relay_params = "relay.params"
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    model = tf.keras.Sequential([
        hub.KerasLayer(tf_hub_links[model_name], output_shape=[1001])
    ])
    img_size = 299 if model_name == 'inceptionv3' else 224
    np_image = np.random.rand(batch_size, img_size, img_size, 3).astype('float32')
    model._set_inputs(np_image)


    # Convert Keras model to ConcreteFunction
    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(
            tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype, name="data"))

    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir="./.tf_saved_model/" + model_name,
                      name="frozen_graph.pb",
                      as_text=False)

    parser = tvm.relay.frontend.TFParser("./.tf_saved_model/"
                                         + model_name +  "/frozen_graph.pb")
    graph_def = parser.parse()
    mod, params = relay.frontend.from_tensorflow(graph_def,
                                                 shape={"data": (batch_size, img_size, img_size, 3)})

    # We assume our model's heavily-layout sensitive operators only consist of nn.conv2d
    desired_layouts = {'nn.conv2d': ['NCHW', 'default']}
    
    # Convert the layout to NCHW
    # RemoveUnunsedFunctions is used to clean up the graph.
    seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
                                    relay.transform.ConvertLayout(desired_layouts)])
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)
    print(mod)

    # with open(relay_file, "w") as fo:
    #     fo.write(tvm.ir.save_json(mod))
    # with open(relay_params, "wb") as fo:
    #     fo.write(relay.save_param_dict(params))

    # with open(relay_file, "r") as fi:
    #     mod = tvm.ir.load_json(fi.read())
    # with open(relay_params, "rb") as fi:
    #     params = relay.load_param_dict(fi.read())
    return mod, params


def main():
    val_path = '/home/ubuntu/tensorflow_datasets/downloads/manual/imagenet2012/val'
    num_calib_samples = 100
    num_test_samples = args.val_samples
    dataset_preparator = DatasetPreparator(val_path, num_calib_samples, num_test_samples, batch_size)
    img_size = 299 if model_name == 'inceptionv3' else 224
    val_dataset = dataset_preparator.preprocess_val(img_size, 'float32')

    # Original 
    fp32_mod, params = get_model()
    compile_and_run(fp32_mod, params, target, "tf_" + model_name + "_fp32", val_dataset, 'data', ctx)
    
    # Data aware 
    calib_dataset = dataset_preparator.preprocess_calib(img_size, 'float32')
    c = prep_calibration_iter(calib_dataset, 'data')
    fp32_mod, params = get_model()
    mod = quantize(fp32_mod, params, True, c)
    compile_and_run(mod, params, target, "tf_" + model_name + "_data", val_dataset, 'data', ctx)


if __name__ == '__main__':
    main()
