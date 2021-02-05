from collections import namedtuple 

BenchAccRecord = namedtuple("BenchAccRecord", ["model", "frontend", "search_strategy", "threshold_estimate_method", "use_per_channel", "hardware", "accuracy"])

def evaluate(frontend, models): 
    records = []
    for use_per_channel in [False, True]:
        for hw_name in ['x86_cpu', 'nvidia_gpu']:
            for model in models:
                if frontend == "mxnet":
                    from test_hago_mxnet import eval_model
                    acc = eval_model(model, use_per_channel, hw_name)
                record = BenchAccRecord(model, frontend, "default", "avg_range", use_per_channel, hw_name, acc)
                records.append(record)
    return records
    
if __name__ == "__main__":
    records = []
    mxnet_models = ['resnet18_v1', 'resnet34_v1', 'resnet50_v1', 'inceptionv3',
                    'vgg16', 'densenet161', 'mobilenet1.0', 'mobilenetv2_1.0',
                    'squeezenet1.1']
    records += evaluate('mxnet', mxnet_models)
    print(records)
