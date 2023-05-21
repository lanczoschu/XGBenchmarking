import yaml
import json
import argparse
from tqdm import tqdm
from pathlib import Path
import torch
import torch.nn as nn
from torch.nn import functional as F
from eval import FidelEvaluation, AUCEvaluation
from get_model import Model
from baselines import LabelPerturb, VGIB, LRIBern, LRIGaussian, CIGA
from utils import to_cpu, log_epoch, get_data_loaders, set_seed, load_checkpoint, ExtractorMLP, get_optimizer
from utils import inherent_models, post_hoc_explainers, post_hoc_attribution, name_mapping
import torchmetrics
from statistics import mean
import numpy as np
import pandas as pd
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")


def eval_one_batch(baseline, data, epoch):
    with torch.set_grad_enabled(baseline.name in ['gradcam', 'gradgeo', 'gnnexplainer']):
        baseline.extractor.eval() if hasattr(baseline, 'extractor') else None
        baseline.clf.eval()

        # do_sampling = True if phase == 'valid' and baseline.name == 'pgexplainer' else False # we find this is better for BernMaskP
        loss, loss_dict, infer_clf_logits, node_attn = baseline.forward_pass(data, epoch=epoch, do_sampling=False)
        # from sklearn.metrics import roc_auc_score
        # roc_auc_score(infer_clf_logits, data.y.cpu())
        return loss_dict, to_cpu(infer_clf_logits), to_cpu(node_attn)


def eval_one_epoch(baseline, data_loader, epoch, phase, seed, signal_class, writer=None, metric_list=None):
    pbar, avg_loss_dict = tqdm(data_loader), dict()
    [eval_metric.reset() for eval_metric in metric_list] if metric_list else None

    clf_ACC, clf_AUC = torchmetrics.Accuracy(task='binary'), torchmetrics.AUROC(task='binary')
    for idx, data in enumerate(pbar):
        # data = negative_augmentation(data, data_config, phase, data_loader, idx, loader_len)
        loss_dict, clf_logits, attn = eval_one_batch(baseline, data.to(baseline.device), epoch)
        ex_labels, clf_labels, data = to_cpu(data.node_label), to_cpu(data.y), data.cpu()
        # print()
        eval_dict = {metric.name: metric.collect_batch(ex_labels, attn, data, signal_class, 'geometric')
                     for metric in metric_list}
        eval_dict.update({'clf_acc': clf_ACC(clf_logits, clf_labels), 'clf_auc': clf_AUC(clf_logits, clf_labels)})
        fidel_list = [eval_dict[k] for k in eval_dict if 'fid' in k and 'all' in k]
        eval_dict.update({'mean_fid': mean(fidel_list) if fidel_list else 0})

        desc = log_epoch(seed, epoch, phase, loss_dict, eval_dict, phase_info='dataset')
        pbar.set_description(desc)
        # compute the avg_loss for the epoch desc
        exec('for k, v in loss_dict.items():\n\tavg_loss_dict[k]=(avg_loss_dict.get(k, 0) * idx + v) / (idx + 1)')

    epoch_dict = {eval_metric.name: eval_metric.eval_epoch() for eval_metric in metric_list} if metric_list else {}
    epoch_dict.update({'clf_acc': clf_ACC.compute().item(), 'clf_auc': clf_AUC.compute().item()})
    epoch_fidel = [epoch_dict[k] for k in epoch_dict if 'fid' and 'all' in k]
    epoch_dict.update({'mean_fid': mean(epoch_fidel) if epoch_fidel else 0})

    # log_epoch(seed, epoch, phase, avg_loss_dict, epoch_dict, writer)

    return epoch_dict

def test(config, method_name, exp_method, model_name, backbone_seed, dataset_name, log_dir, device):
    set_seed(backbone_seed)
    writer = None
    print('The logging directory is', log_dir), print('=' * 80)

    batch_size = config['optimizer']['batch_size']
    data_config = config['data']
    loaders, test_set, dataset = get_data_loaders(dataset_name, batch_size, data_config, dataset_seed=0)
    signal_class = dataset.signal_class

    clf = Model(model_name, config[model_name],  # backbone_config
                method_name, config[method_name],  # method_config
                dataset).to(device)

    extractor = ExtractorMLP(config[model_name]['hidden_size'], config[method_name], config['data'].get('use_lig_info', False)) \
        if method_name in inherent_models + ['pgexplainer'] else nn.Identity()
    extractor = extractor.to(device)
    criterion = F.binary_cross_entropy_with_logits

    # establish the model and the metrics

    backbone = eval(name_mapping[method_name])(clf, extractor, criterion, config[method_name]) \
        if method_name in inherent_models else clf
    assert load_checkpoint(backbone, log_dir, model_name=method_name, seed=backbone_seed, map_location=torch.device('cpu') if not torch.cuda.is_available() else None)

    if 'label_perturb' in exp_method:
        model = LabelPerturb(backbone, mode=int(exp_method[-1]))
    else:
        model = eval(name_mapping[exp_method])(clf, criterion, config[exp_method])

    # metric_list = []
    # metric_list = [AUCEvaluation()] + [FidelEvaluation(backbone, i/10) for i in range(2, 9)] + \
    #  [FidelEvaluation(backbone, i/10, instance='pos') for i in range(2, 9)] + \
    metric_list = [FidelEvaluation(backbone, 0.5*i/10 + 0.8, instance='neg') for i in reversed(range(5))]
    print('Use random explanation and fidelity w/ signal nodes to test the Model Sensitivity.')

    # metric_names = [i.name for i in metric_list] + ['clf_acc', 'clf_auc']
    # train_dict = eval_one_epoch(model, loaders['train'], 1, 'train', backbone_seed, signal_class, writer, metric_list)
    # valid_dict = eval_one_epoch(model, loaders['valid'], 1, 'valid', backbone_seed, signal_class, writer, metric_list)
    test_dict = eval_one_epoch(model, loaders['test'], 1, 'test', backbone_seed, signal_class,  writer, metric_list)

    return {}, {}, test_dict

def save_multi_bseeds(multi_methods_res, method_name, exp_method, seeds):
    seeds += ['avg', 'std']
    indexes = [method_name+'_'+str(seed) for seed in seeds]
    # from itertools import product
    # indexes = ['_'.join(item) for item in product(methods, seeds)]
    df = pd.DataFrame(multi_methods_res, index=indexes)

    day_dir = Path('result') / config_name / 'sensitivity' / datetime.now().strftime("%m_%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = day_dir / ('_'.join([exp_method, method_name, 'sensitivity.csv']))
    with open(csv_dir, mode='w') as f:
        df.to_csv(f, lineterminator="\n", header=f.tell() == 0)
    return df

def get_avg_std_report(reports):
    # print(reports)
    all_keys = {k: [] for k in reports[0]}
    for report in reports:
        for k in report:
            all_keys[k].append(report[k])
    avg_report = {k: np.mean(v) for k, v in all_keys.items()}
    std_report = {k: np.std(v) for k, v in all_keys.items()}
    avg_std_report = {k: f'{np.mean(v):.3f} +/- {np.std(v):.3f}' for k, v in all_keys.items()}
    return avg_report, std_report, avg_std_report

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EXP')
    parser.add_argument('-d', '--dataset', type=str, help='dataset used', default='actstrack')
    parser.add_argument('--clf_method', type=str, help='method used', default='erm', choices=inherent_models+['erm'])
    parser.add_argument('--exp_method', type=str, help='method used', default='label_perturb1', choices=['label_perturb1', 'label_perturb0'])
    parser.add_argument('-b', '--backbone', type=str, help='backbone used', default='egnn')
    parser.add_argument('--cuda', type=int, help='cuda device id, -1 for cpu', default=-1)
    parser.add_argument('--note', type=str, help='note in log name', default='')
    parser.add_argument('--bseeds', type=int, nargs="+", help='random seed for training backbone', default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # parser.add_argument('--mseed', type=int, help='random seed for explainer', default=0)

    args = parser.parse_args()
    # sub_metric = 'avg_loss'
    print(args)

    dataset_name, method_name, model_name, cuda_id, note = args.dataset, args.clf_method, args.backbone, args.cuda, args.note
    exp_method = args.exp_method
    config_name = '_'.join([model_name, dataset_name])
    # sub_dataset_name = '_' + dataset_name.split('_')[1] if len(dataset_name.split('_')) > 1 else ''
    config_path = Path('./configs') / f'{config_name}.yml'
    config = yaml.safe_load((config_path).open('r'))
    if config[method_name].get(model_name, False):
        config[method_name].update(config[method_name][model_name])

    print('=' * 80)
    print(f'Config for {method_name}: ', json.dumps(config[method_name], indent=4))



    # the directory is like egnn_actstrack / bseed0 / lri_bern
    multi_seeds_res = []
    for backbone_seed in args.bseeds:
        cuda_id = backbone_seed % 5
        device = torch.device(f'cuda:{cuda_id}' if cuda_id >= 0 and torch.cuda.is_available() else 'cpu')
        model_dir = Path('log') / config_name / method_name
        train_report, valid_report, test_report = test(config, method_name, exp_method, model_name, backbone_seed, dataset_name, model_dir, device)
        multi_seeds_res += [test_report]
        # print('Train Dataset Result: ', json.dumps(train_report, indent=4))
        # print('Valid Dataset Result: ', json.dumps(valid_report, indent=4))
        print('Test Dataset Result: ', json.dumps(test_report, indent=4))
    avg_report, std_report, avg_std_report = get_avg_std_report(multi_seeds_res)
    multi_seeds_res += [avg_report, std_report]
    print(json.dumps(avg_std_report, indent=4))
    save_multi_bseeds(multi_seeds_res, method_name, exp_method, args.bseeds)

