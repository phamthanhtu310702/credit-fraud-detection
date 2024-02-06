import torch

from sklearn.metrics import average_precision_score


def get_node_classification_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the node classification task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.cpu().detach().numpy().flatten()
    labels = labels.cpu().numpy().flatten()

    ap = average_precision_score(y_true=labels, y_score = predicts, pos_label= 0)

    return {'average_precision_score': ap}
