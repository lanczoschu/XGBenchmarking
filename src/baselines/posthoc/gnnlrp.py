import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch_scatter import scatter, scatter_max
from backbones import GINConv
# from evaluation import control_sparsity
from ..base import BaseRandom


class GNNLRP(BaseRandom):

    def __init__(self, clf, criterion, config):
        super().__init__()
        self.name = 'gnnlrp'
        self.clf = clf
        self.target_layers = clf.model.convs
        n_layers = len(self.target_layers)
        self.gammas = config.get("gammas", np.linspace(3, 0, n_layers-1))

        self.criterion = criterion
        self.device = next(self.parameters()).device

    def start_tracking(self):
        self.activations_and_grads = Transforms(self.clf, self.target_layers)

    def forward_pass(self, data, epoch, do_sampling):
        original_clf_logits = self.activations_and_grads(data)
        # H_end, transforms = self.get_H_transforms(data, gammas)
        H_end = self.activations_and_grads.activations[-1].data
        sum_nodes = 0
        node_weights_list = []
        for graph in data.to_data_list():
            add_nodes = graph.num_nodes
            relevance = H_end[sum_nodes:sum_nodes+add_nodes, :]
            transforms = self.get_single_graph_transforms(graph, self.gammas, start=sum_nodes)
            sum_nodes = sum_nodes+add_nodes
            for transform in reversed(transforms):
                # einsum slow
                # relevance_subgraph = torch.einsum('ijkl,kl->ij', transform, mask @ relevance_subgraph)
                nbnodes = transform.shape[0]
                nbneurons_in = transform.shape[1]
                nbneurons_out = transform.shape[3]

                transform = transform.reshape(nbnodes * nbneurons_in, nbnodes * nbneurons_out)
                relevance = relevance.reshape([nbnodes * nbneurons_out, 1])

                relevance = (transform @ relevance).reshape(nbnodes, nbneurons_in)
            graph_node_weights = relevance.sum(axis=1)
            node_weights_list.append(graph_node_weights)
        node_weights = torch.cat(node_weights_list)

        res_weights = self.node_attn_to_edge_attn(node_weights, data.edge_index) if hasattr(data, 'edge_label') else node_weights
        # sparse_edge_mask = control_sparsity(edge_attn)
        # sparsity_masked_logits = self.clf(data, edge_attn=sparse_edge_mask)

        return -1, {}, original_clf_logits, res_weights.sigmoid()

    @staticmethod
    def get_adj(data):
        adj = torch.eye(data.num_nodes, device=data.edge_index.device)
        for i, j in data.edge_index.T:
            adj[i, j] = 1
            adj[j, i] = 1
        return adj

    def get_single_graph_transforms(self, graph, gammas, start):
        activations_list = [a.data for a in self.activations_and_grads.activations]
        weight_list = [g.data for g in self.activations_and_grads.weights]
        bias_list = [b.data for b in self.activations_and_grads.biases]
        transforms = []
        A = self.get_adj(graph)
        for W, b, H, gamma in zip(weight_list, bias_list, activations_list, gammas):
            W = W + gamma * W.clamp(min=0)
            b = b + gamma * b.clamp(min=0) + 1e-6
            H = H[start:start+graph.num_nodes, :]
            neuron_weight = (A.unsqueeze(-1).unsqueeze(-1) * W.unsqueeze(0).unsqueeze(0)).permute(0, 3, 1, 2)
            neuro_feature = H.unsqueeze(0).unsqueeze(0)
            transform = neuron_weight * neuro_feature  # 这里没加b？
            # focus on the computation here especially the 'b'
            transform = transform / (transform.sum(axis=0).sum(axis=0) + b).unsqueeze(0).unsqueeze(0)
            transforms.append(transform)
        return transforms

    def get_H_transforms(self, data, gammas):
        start = 0
        activations_list = [a.cpu().data for a in self.activations_and_grads.activations]
        weight_list = [g.cpu().data for g in self.activations_and_grads.weights]
        bias_list = [b.cpu().data for b in self.activations_and_grads.biases]
        transforms = []

        for W, b, H, gamma in zip(weight_list, bias_list, activations_list, gammas):
            W = W + gamma * W.clamp(min=0)
            b = b + gamma * b.clamp(min=0)
            batch_transforms = []
            start = 0
            for i in range(data.num_graphs):
                graph = data.get_example(i)
                A = self.get_adj(graph)
                neuron_weight = (A.unsqueeze(-1).unsqueeze(-1) * W.unsqueeze(0).unsqueeze(0)).permute(0, 3, 1, 2)
                neuro_feature = H[start:start+graph.num_nodes, :].unsqueeze(0).unsqueeze(0)
                transform = neuron_weight*neuro_feature # 这里没加b？
                transform = transform / (transform.sum(axis=-1).sum(axis=-1)+b).unsqueeze(-1).unsqueeze(-1)
                start = start + graph.num_nodes
                batch_transforms.append(transform)

            batch_transform = self.to_batch_trans(batch_transforms, node_size=data.num_nodes)
            transforms.append(batch_transform)


        return activations_list[-1], transforms

    @staticmethod
    def to_batch_trans(batch_trans, node_size):
        hidden_size = batch_trans[0].shape[1]
        res = torch.zeros([node_size, node_size, hidden_size, hidden_size])
        num_nodes = 0
        for four_dim_tensor in batch_trans:
            four_dim_tensor.permute(0, 2, 1, 3)
            add_nodes = four_dim_tensor.shape[0]
            res[num_nodes:num_nodes+add_nodes, num_nodes:num_nodes+add_nodes, :, :] = four_dim_tensor
            num_nodes = num_nodes + add_nodes
        return res

    @staticmethod
    def node_attn_to_edge_attn(node_attn, edge_index):
        src_attn = node_attn[edge_index[0]]
        dst_attn = node_attn[edge_index[1]]
        edge_attn = (src_attn + dst_attn) / 2
        return edge_attn

class Transforms:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform=None):
        self.model = model
        self.activations = []
        self.weights = []
        self.biases = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation))
            # Because of https://github.com/pytorch/pytorch/issues/61519,
            # we don't use backward hook to record gradients.
            # self.handles.append(
            #     target_layer.register_forward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        if isinstance(output, tuple):  # egnn outputs a tuple
            output = output[0]

        reg_fee_dic = RegexMap(module.state_dict(), None)
        activation = output
        weight = reg_fee_dic['weight']
        bias = reg_fee_dic['bias']

        for mod in module.children():
            if isinstance(mod, GINConv):
                weight = torch.ones_like(weight)
        # weight = torch.ones_like(weight)
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.weights.append(weight.detach())
        self.biases.append(bias.detach())
        self.activations.append(activation.detach())

    def save_gradient(self, module, input, output):
        if isinstance(output, tuple):  # egnn outputs a tuple
            output = output[0]

        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on tensor requires grad.
            return

        # Gradients are computed in reverse order
        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients = [grad.detach()] + self.gradients

        output.register_hook(_store_grad)

    def __call__(self, x):
        self.weights = []
        self.activations = []
        self.biases = []
        return self.model(x)

    def release(self):
        for handle in self.handles:
            handle.remove()

from re import search, I
class RegexMap:
    def __init__(self, n_dic, val):
        self._items = n_dic
        self.__val = val

    def __getitem__(self, key):
        for regex in reversed(self._items.keys()):
            if search(key, regex, I):
                return self._items[regex]
        return self.__val