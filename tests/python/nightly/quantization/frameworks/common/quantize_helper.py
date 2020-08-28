import tvm
from tvm import relay
from tvm import hago
import logging
logging.basicConfig(level=logging.DEBUG)

def quantize(mod, params, data_aware, iterator):
    return quantize_hago(mod, params, data_aware, iterator)


def quantize_hago(mod, params, data_aware, iterator):

    qconfig = hago.qconfig(skip_conv_layers=[0],
                           log_file='temp.log')

    with qconfig:
        graph = hago.prerequisite_optimize(mod['main'], params=params)
        logging.debug('current quantize config')
        logging.debug(hago.current_qconfig())
        hardware = hago.create_accelerator_description()
        space = hago.generate_search_space(graph, hardware)
        # tuner = hago.BatchedGreedySearchTuner(space, 'accuracy')
        tuner = hago.DefaultSetting(space, 'accuracy')
        ctx = tvm.cpu()
        target = 'llvm'
        strategy, result = hago.search_quantize_strategy(graph, hardware, iterator, tuner, ctx, target)

        quantizer = hago.create_quantizer(graph, hardware, strategy)
        simulated_graph = quantizer.simulate()
        quantized_graph = quantizer.quantize()
        logging.debug('simulated graph')
        logging.debug(simulated_graph.astext(show_meta_data=False))
        logging.debug('quantize graph')
        logging.debug(quantized_graph.astext(show_meta_data=False))
        # hago.inspect_graph_statistic(graph, hardware, strategy, dataset, ctx, target)
        return tvm.IRModule.from_expr(quantized_graph)


def quantize_old_tvm(mod, params, data_aware, iterator):
    if data_aware:
        with relay.quantize.qconfig(calibrate_mode='avg_min_max', weight_scale='max',
                skip_conv_layers=[0]):
            mod = relay.quantize.quantize(mod, params, dataset=iterator)
    else:
        with relay.quantize.qconfig(calibrate_mode='global_scale', global_scale=7.9):
            mod = relay.quantize.quantize(mod, params)
    return mod

