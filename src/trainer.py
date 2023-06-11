import os
pre_cuda_id = int(os.environ.get('CUDA_VISIBLE_DEVICES')) if os.environ.get('CUDA_VISIBLE_DEVICES') else None
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5'
import yaml
import json
import pandas as pd
import shutil
import argparse
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from itertools import product
import nni
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from eval import FidelEvaluation, LabelFidelity, AUCEvaluation
from get_model import Model
from baselines import *
from utils import to_cpu, log_epoch, get_data_loaders, set_seed, update_and_save_best_epoch_res, load_checkpoint, save_checkpoint, ExtractorMLP, get_optimizer
from utils import inherent_models, post_hoc_explainers, post_hoc_attribution, name_mapping
import torchmetrics
from statistics import mean
import warnings
warnings.filterwarnings("ignore")


def negative_augmentation(data, data_config, phase, data_loader, batch_idx, loader_len):
    # only used for PLBind dataset
    if data_config['data_name'] == 'plbind':
        neg_aug_p = data_config['neg_aug_p']
        if neg_aug_p and np.random.rand() < neg_aug_p and batch_idx != loader_len - 1 and phase == 'train':
            aug_data = next(iter(data_loader))
            data.x_lig = aug_data.x_lig
            data.pos_lig = aug_data.pos_lig
            data.x_lig_batch = aug_data.x_lig_batch
            data.y = torch.zeros_like(data.y)
    return data


def eval_one_batch(baseline, optimizer, data, epoch, phase):
    with torch.set_grad_enabled(baseline.name in ['gradcam', 'gradx', 'inter_grad', 'gnnexplainer']):
        assert optimizer is None
        baseline.extractor.eval() if hasattr(baseline, 'extractor') else None
        baseline.clf.eval()

        # BernMaskP
        do_sampling = True if phase == 'valid' and baseline.name == 'pgexplainer' else False # we find this is better for BernMaskP
        loss, loss_dict, infer_clf_logits, node_attn = baseline.forward_pass(data, epoch=epoch, do_sampling=do_sampling)

        return loss_dict, to_cpu(infer_clf_logits), to_cpu(node_attn)


def train_one_batch(baseline, optimizer, data, epoch, phase):
    baseline.extractor.train() if hasattr(baseline, 'extractor') else None
    baseline.clf.train() if (baseline.name != 'pgexplainer' or phase == 'warm') else baseline.clf.eval()

    if phase == 'warm':
        loss, loss_dict, org_clf_logits, node_attn = baseline.warming(data)
    else:
        loss, loss_dict, org_clf_logits, node_attn = baseline.forward_pass(data, epoch=epoch, do_sampling=True)
    optimizer.zero_grad()

    if baseline.name == 'test_inherent':
        loss.backward(retain_graph=True)
        data.node_grads += data.pos.grad.norm(dim=1, p=2)
        data.edge_grads += data.edge_attn.grad
    else:
        loss.backward()

    optimizer.step()
    return loss_dict, to_cpu(org_clf_logits), to_cpu(node_attn)


def run_one_epoch(baseline, optimizer, data_loader, epoch, phase, seed, signal_class, writer=None, metric_list=None, return_attn=False):
    use_tqdm = True
    run_one_batch = train_one_batch if optimizer else eval_one_batch
    pbar, avg_loss_dict = tqdm(data_loader) if use_tqdm else data_loader, dict()
    [eval_metric.reset() for eval_metric in metric_list] if phase in ['valid', 'test'] else None

    clf_ACC, clf_AUC = torchmetrics.Accuracy(task='binary'), torchmetrics.AUROC(task='binary')
    save_epoch_attn = []
    for idx, data in enumerate(pbar):
        # data = negative_augmentation(data, data_config, phase, data_loader, idx, loader_len)
        loss_dict, clf_logits, attn = run_one_batch(baseline, optimizer, data.to(baseline.device), epoch, phase)
        ex_labels, clf_labels, data = to_cpu(data.node_label), to_cpu(data.y), data.cpu()

        # prepare to save attn
        if return_attn:
            graph_labels = data.y[data.batch]
            batch_idx = torch.full_like(graph_labels, idx)
            graph_idx = data.batch.unsqueeze(-1)
            save_attn = torch.cat([attn.unsqueeze(-1), ex_labels.unsqueeze(-1), graph_labels, batch_idx, graph_idx], dim=1)
            save_epoch_attn.append(save_attn)

        eval_dict = {metric.name: metric.collect_batch(ex_labels, attn, data, signal_class, 'geometric')
                     for metric in metric_list} if phase in ['valid', 'test'] else {}
        eval_dict.update({'clf_acc': clf_ACC(clf_logits, clf_labels), 'clf_auc': clf_AUC(clf_logits, clf_labels)})
        batch_fid = [eval_dict[k] for k in eval_dict if 'fid' in k and 'all' in k]
        eval_dict.update({'mean_fid': mean(batch_fid)}) if phase in ['valid', 'test'] and batch_fid else {}

        desc = log_epoch(seed, epoch, phase, loss_dict, eval_dict)
        pbar.set_description(desc) if use_tqdm else None
        # compute the avg_loss for the epoch desc
        exec('for k, v in loss_dict.items():\n\tavg_loss_dict[k]=(avg_loss_dict.get(k, 0) * idx + v) / (idx + 1)')

    epoch_dict = {eval_metric.name: eval_metric.eval_epoch() for eval_metric in metric_list} if metric_list else {}
    epoch_dict.update({'clf_acc': clf_ACC.compute().item(), 'clf_auc': clf_AUC.compute().item()})
    fid_score = [epoch_dict[k] for k in epoch_dict if 'fid' in k and 'all' in k]
    epoch_dict.update({'mean_fid': mean(fid_score)}) if phase in ['valid', 'test'] and fid_score else {}

    log_epoch(seed, epoch, phase, avg_loss_dict, epoch_dict, writer)

    if return_attn:
        save_epoch_attn = torch.cat(save_epoch_attn, dim=0)
        return epoch_dict, save_epoch_attn
    else:
        return epoch_dict


def train(config, method_name, model_name, backbone_seed, seed, dataset_name, parent_dir, device, main_metric, quick=False, save=False):
    # writer = SummaryWriter(log_dir) if log_dir is not None else None
    writer = None
    model_dir, log_dir = (parent_dir / method_name, ) * 2 if method_name in inherent_models \
        else (parent_dir / 'erm', None) if model_name in post_hoc_attribution \
        else (parent_dir / 'erm', parent_dir / method_name)
    # log_dir = parent_dir / method_name if method_name in inherent_models else None
    log_dir.mkdir(parents=True, exist_ok=True) if log_dir is not None else None
    model_dir.mkdir(parents=True, exist_ok=True)

    batch_size = config['optimizer']['batch_size']
    model_cofig = config[method_name] if method_name in inherent_models else config['erm']
    warmup = model_cofig['warmup']
    epochs = config[method_name]['epochs']
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
    constructor = eval(name_mapping[method_name])
    optimizer = get_optimizer(clf, extractor, config['optimizer'], method_name, warmup=True)

    # establish the model and the metrics
    if method_name in inherent_models:
        baseline = constructor(clf, extractor, criterion, config[method_name])
        for epoch in range(1, warmup+1):
            run_one_epoch(baseline, optimizer, loaders['train'], epoch, 'warm', seed, signal_class, writer)
            if save:
                save_checkpoint(baseline.clf, model_dir, model_name='erm', backbone_seed=backbone_seed,
                                seed=backbone_seed)
        metric_list = [AUCEvaluation()]
    elif method_name in post_hoc_attribution + post_hoc_explainers:
        baseline = constructor(clf, criterion, config[method_name]) if method_name != 'pgexplainer' else PGExplainer(clf, extractor, criterion, config['pgexplainer'])
        if not load_checkpoint(baseline.clf, model_dir, model_name='erm', seed=backbone_seed, map_location=torch.device('cpu') if not torch.cuda.is_available() else None):
            for epoch in range(1, warmup + 1):
                run_one_epoch(baseline, optimizer, loaders['train'], epoch, 'warm', backbone_seed, signal_class, writer)
            save_checkpoint(baseline.clf, model_dir, model_name='erm', backbone_seed=backbone_seed, seed=backbone_seed)
        metric_list = [AUCEvaluation()] + [FidelEvaluation(baseline.clf, i/10) for i in range(2, 9)] + \
                  [FidelEvaluation(baseline.clf, i/10, instance='pos') for i in range(2, 9)] + \
                  [FidelEvaluation(baseline.clf, i/10, instance='neg') for i in range(2, 9)] if quick==False else \
                  [AUCEvaluation()] + [FidelEvaluation(baseline.clf, i/10) for i in range(2, 9)]
        baseline.start_tracking() if 'grad' in method_name or method_name == 'gnnlrp' else None
    else:
        assert 'test' == method_name
        baseline = Test(clf, criterion, config=None)
        load_checkpoint(baseline.clf, model_dir, model_name='erm', seed=backbone_seed, map_location=torch.device('cpu') if not torch.cuda.is_available() else None)
        # for epoch in range(1, warmup+1):
        #     run_one_epoch(baseline, optimizer, loaders['train'], epoch, 'warm', seed, signal_class, writer)
        metric_list = [AUCEvaluation()]
        print('New method is ready!')

    set_seed(seed)
    metric_names = [a + b for a, b in product(['valid_', 'test_'], [i.name for i in metric_list]+['clf_acc', 'clf_auc', 'mean_fid'])]
    # metric_names = [j+i.name for i in metric_list for j in ['valid_', 'test_']]
    metric_dict = {}.fromkeys(metric_names, 0)
    best_attn = None
    optimizer = get_optimizer(clf, extractor, config['optimizer'], method_name, warmup=False)
    for epoch in range(1, epochs+1):
        if method_name in inherent_models + ['pgexplainer']:
            run_one_epoch(baseline, optimizer, loaders['train'], epoch, 'train', seed, signal_class, writer, metric_list)
            valid_dict = run_one_epoch(baseline, None, loaders['valid'], epoch, 'valid', seed, signal_class, writer, metric_list)
            test_dict, epoch_attn = run_one_epoch(baseline, None, loaders['test'], epoch, 'test', seed,
                                                  signal_class, writer, metric_list, return_attn=True)
        else:
            test_dict, epoch_attn = run_one_epoch(baseline, None, loaders['test'], epoch, 'test', seed,
                                                  signal_class,  writer, metric_list, return_attn=True)
            valid_dict = test_dict # other methods don't need validation to select epochs

        # print(metric_dict)
        metric_dict, new_best = update_and_save_best_epoch_res(baseline, metric_dict, valid_dict, test_dict, epoch, log_dir, backbone_seed, seed, writer, method_name, main_metric)
        best_attn = epoch_attn if new_best else best_attn
        metric_dict.update({'default': metric_dict[f'valid_{main_metric}']})
        nni.report_intermediate_result(metric_dict)

    meta_index = 'attn' if method_name in post_hoc_attribution + inherent_models else seed
    indexes = [meta_index, 'node_labels', 'graph_labels', 'batch_idx', 'graph_idx']

    return metric_dict, (best_attn, indexes)


def run_one_seed(args, optimized_params):
    print(args)
    dataset_name, method_name, model_name, cuda_id, note = args.dataset, args.method, args.backbone, args.cuda, args.note
    method_seed, backbone_seed = args.seed, args.bseed
    main_metric = 'clf_auc' if method_name in inherent_models + ['test'] else 'mean_fid'
    set_seed(backbone_seed)
    config_name = '_'.join([model_name, dataset_name])
    # sub_dataset_name = '_' + dataset_name.split('_')[1] if len(dataset_name.split('_')) > 1 else ''
    config_path = Path('./configs') / f'{config_name}.yml'
    config = yaml.safe_load((config_path).open('r'))
    if config[method_name].get(model_name, False):
        config[method_name].update(config[method_name][model_name])
    if optimized_params:
        config[method_name].update(optimized_params)
    print('=' * 80)
    print(f'Config for {method_name}: ', json.dumps(config[method_name], indent=4))
    # :{pre_cuda_id}
    cuda_ = f'cuda' if cuda_id == 99 else f'cuda:{cuda_id}' if cuda_id >= 0 else 'cpu'
    device = torch.device(cuda_)
    # log_dir = None
    # if config['logging']['tensorboard'] or method_name in ['gradcam', 'gradgeo', 'bernmask_p', 'bernmask']:
    main_dir = Path('log') / config_name
    main_dir.mkdir(parents=True, exist_ok=True)
        # shutil.copy(config_path, log_dir / config_path.name)
    report_dict, (best_attn, indexes) = train(config, method_name, model_name, backbone_seed, method_seed, dataset_name, main_dir, device, main_metric, quick=args.quick, save=args.save)
    attn_df = pd.DataFrame(best_attn, columns=indexes)

    return report_dict, attn_df


def main(args):
    if args.gpu_ratio is not None:
        torch.cuda.set_per_process_memory_fraction(args.gpu_ratio)

    params = nni.get_next_parameter()
    report_dict, attn_df = run_one_seed(args, params)

    print(json.dumps(report_dict, indent=4))
    # nni_report = dict(report_dict, **{'default': report_dict[f'valid_{main_metric}']})
    nni.report_final_result(report_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EXP')
    parser.add_argument('-d', '--dataset', type=str, help='dataset used', default='actstrack_2T')
    parser.add_argument('-m', '--method', type=str, help='method used', default='lri_gaussian')
    parser.add_argument('-b', '--backbone', type=str, help='backbone used', default='egnn')
    parser.add_argument('--cuda', type=int, help='cuda device id, -1 for cpu', default=-1)
    parser.add_argument('--seed', type=int, help='random seed', default=0)
    parser.add_argument('--note', type=str, help='note in log name', default='')
    parser.add_argument('--gpu_ratio', type=float, help='gpu memory ratio', default=None)
    parser.add_argument('--bseed', type=int, help='random seed for training backbone', default=0)
    parser.add_argument('--quick', action="store_true", help='ignore some evaluation')
    parser.add_argument('--no_tqdm', action="store_true", help='disable the tqdm')

    exp_args = parser.parse_args()
    use_tqdm = False if exp_args.no_tqdm else True
    # sub_metric = 'avg_loss'
    main(exp_args)
