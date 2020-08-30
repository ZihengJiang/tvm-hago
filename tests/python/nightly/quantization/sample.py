import tvm
from tvm import relay
from tvm import hago
import numpy as np

def create_hardware():
    hardware = hago.Hardware()
    hardware.add_op_desc('concatenate', hago.OpDesc(in_dtypes='int8', out_dtypes='int8'))
    hardware.add_op_desc('nn.dense', hago.OpDesc(in_dtypes='int8', out_dtypes='int32'))
    return hardware

def target_and_ctx(device):
    if device == 'cpu':
        target = 'llvm'
        ctx = tvm.cpu()
    elif device == 'gpu':
        target = 'cuda'
        ctx = tvm.gpu(1)
    return target, ctx

def test_dense(ishape=(32, 16), wshape=(10, 16), batch_num=3, device='cpu'):
    target, ctx = target_and_ctx(device)
    data = relay.var('data', shape=ishape)
    weight = relay.var('weight', shape=wshape)
    out = relay.nn.dense(data, weight)
    func = relay.Function([data, weight], out)

    weight_np = np.random.rand(*wshape).astype('float32')

    # generate dataset
    batches = []
    for i in range(batch_num):
        data_np = np.random.rand(*ishape).astype('float32')
        ex = relay.create_executor("debug", ctx=ctx, target=target)
        out_np = ex.evaluate(func)(data_np, weight_np).asnumpy()
        pred_np = np.argmax(out_np, axis=1)
        batches.append({'data': tvm.nd.array(data_np), 'label': tvm.nd.array(pred_np)})
    dataset = hago.CalibrationDataset(batches)

    params = {'weight': tvm.nd.array(weight_np)}
    return func, params, dataset


def test_concatenate(ishape=(8, 16), wshape=(10, 16), batch_num=3, device='cpu'):
    target, ctx = target_and_ctx(device)
    data_a = relay.var('data_a', shape=ishape)
    data_b = relay.var('data_b', shape=ishape)
    data_c = relay.var('data_c', shape=ishape)
    data_d = relay.var('data_d', shape=ishape)
    data = relay.concatenate([data_a, data_b, data_c, data_d], axis=0)
    weight = relay.var('weight', shape=wshape)
    out = relay.nn.dense(data, weight)
    func = relay.Function([data_a, data_b, data_c, data_d, weight], out)

    weight_np = np.random.rand(*wshape).astype('float32')

    # generate dataset
    batches = []
    for i in range(batch_num):
        data_a_np = np.random.rand(*ishape).astype('float32')
        data_b_np = np.random.rand(*ishape).astype('float32')
        data_c_np = np.random.rand(*ishape).astype('float32')
        data_d_np = np.random.rand(*ishape).astype('float32')
        ex = relay.create_executor("debug", ctx=ctx, target=target)
        out_np = ex.evaluate(func)(data_a_np, data_b_np, data_c_np, data_d_np, weight_np).asnumpy()
        pred_np = np.argmax(out_np, axis=1)
        batches.append({'data_a': tvm.nd.array(data_a_np),
                        'data_b': tvm.nd.array(data_b_np),
                        'data_c': tvm.nd.array(data_c_np),
                        'data_d': tvm.nd.array(data_d_np),
                        'label': tvm.nd.array(pred_np)})
    dataset = hago.CalibrationDataset(batches)

    params = {'weight': tvm.nd.array(weight_np)}
    return func, params, dataset


def check_results(func, params, dataset, device='cpu'):
    target, ctx = target_and_ctx(device)
    # prepared by user
    hardware = create_hardware()
    
    qconfig = hago.qconfig(skip_conv_layers=[0],
                           log_file='temp.log',
                           threshold_estimate_method="power_of_two_range")
    with qconfig:
        func = hago.prerequisite_optimize(func, params)
        print('after optimize')
        print(func)
        space = hago.generate_search_space(func, hardware)
        # tuner = hago.BatchedGreedySearchTuner(space, 'accuracy')
        tuner = hago.DefaultSetting(space, 'accuracy')
        strategy, result = hago.search_quantize_strategy(func, hardware, dataset, tuner, ctx, target)
        quantizer = hago.create_quantizer(func, hardware, strategy)
        simulated_graph = quantizer.simulate()
        quantized_graph = quantizer.quantize()
        print(strategy)
        print(result)
        print(simulated_graph)
        print(quantized_graph)

if __name__ == '__main__':
    device = 'cpu'
    func, params, dataset = test_concatenate(device=device)
    print('original model:')
    print(func)
    check_results(func, params, dataset, device)
