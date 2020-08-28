""" Accuracy measurement. """

class AccuracyAggregator(object):
    def __init__(self):
        self.top1 = 0
        self.top5 = 0
        self.images = 0

    def is_top1(self, tensor, gt):
        return 1 if tensor[0] == gt else 0

    def is_top5(self, tensor, gt):
        return 1 if gt in tensor else 0

    def update(self, gt, tensor):
        batch_size = tensor.shape[0]
        for idx in range(batch_size):
            self.top1 += self.is_top1(tensor[idx], gt[idx])
            self.top5 += self.is_top5(tensor[idx], gt[idx])
            self.images += 1

    def report(self):
        top1 = round(self.top1 * 100.0/self.images, 2)
        top5 = round(self.top5 * 100.0/self.images, 2)
        return (top1, top5)
