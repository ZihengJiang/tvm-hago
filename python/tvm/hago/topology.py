# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
from __future__ import absolute_import

from .base import *
from .hardware import *

from tvm import relay
from collections import OrderedDict, namedtuple

# make topology reference unknown

class NodeKind:
    Input = 1
    Weight = 2
    Activation = 3

class Topology(object):
    def __init__(self, graph):
        self.graph = graph
        self.hardware = None
        self._node2idx = self._build_node_index()
        self._edge2idx = self._build_edge_index()
        self._node2edges = self._build_node2edges()
        self._node2kind = self._build_node2kind()
        self._node2layout, self._node2channel_axis = self._build_node2layoutinfo()

    def is_quantized_node(self, node):
        assert self.hardware is not None
        return self._node_conds[self._node2idx[node]]

    def is_quantized_edge(self, edge):
        assert self.hardware is not None
        return self._edge_conds[self._edge2idx[edge]]

    def node2idx(self):
        return self._node2idx

    def edge2idx(self):
        return self._edge2idx

    def node2edges(self):
        return self._node2edges

    def node2kind(self):
        return self._node2kind

    def node2layout(self):
        return self._node2layout

    def node2channel_axis(self):
        return self._node2channel_axis

    def analyze(self, hardware):
        self.hardware = hardware
        node2idx = self.node2idx()
        for node in node2idx:
            print(node_str(node))
        edge2idx = self.edge2idx()
        for edge in edge2idx:
            print(edge_str(edge))
        # expand condition list
        self._node_conds = [None] * len(node2idx)
        self._edge_conds = [None] * len(edge2idx)

        def fvisit_analyze(node):
            def set_cond(node, cond):
                nidx = node2idx[node]
                self._node_conds[nidx] = cond
                for edge in list_in_edges(node):
                    eidx = edge2idx[edge]
                    self._edge_conds[eidx] = cond

            if isinstance(node, (relay.Var, relay.Constant)):
                # mark variable as float
                self._node_conds[node2idx[node]] = False
                return

            if isinstance(node, relay.Call):
                # print(node.op.name)
                if not hardware.list_integer_descs(node):
                    # current op does not support integer computation 
                    set_cond(node, False)
                    return

                src_node_conds = [self._node_conds[node2idx[src]] for src in list_in_nodes(node)]
                if not any(src_node_conds) and hardware.list_float_descs(node):
                    # all float input and current op support float computation
                    set_cond(node, False)
                else:
                    set_cond(node, True)
                return
        relay.analysis.post_order_visit(self.graph, fvisit_analyze)

        print('analyzed condition')
        print('node_conds: {}'.format(self._node_conds))
        print('edge_conds: {}'.format(self._edge_conds))

        # check that all condition has been set properly
        for cond in self._node_conds + self._edge_conds:
            assert cond is not None

    def generate_search_space(self):
        assert self.hardware is not None
        hardware = self.hardware
        node2idx = self.node2idx()
        edge2idx = self.edge2idx()

        # generate a maximum bit list, whose order is same with edge2idx
        # but without non-quantized edges
        bits = []
        for node in node2idx:
            if self.is_quantized_node(node):
                for src_idx, src in enumerate(list_in_nodes(node)):
                    dst_can_consume = [desc.in_dtype(src_idx).bits for desc in hardware.list_integer_descs(node)]
                    if isinstance(src, (relay.Var, relay.Constant)):
                        src_can_produce = []
                    else:
                        src_can_produce = [desc.out_dtype(0).bits for desc in hardware.list_integer_descs(src)]
                    max_consume = max(dst_can_consume) if len(dst_can_consume) else None
                    max_produce = max(src_can_produce) if len(src_can_produce) else None
                    final_bit = min_with_none(max_consume, max_produce)
                    bits.append(final_bit)

        print('bit limit')
        print(bits)
        self.edge2bit = self.build_edge_info(bits)
        self.print_edge_info(self.edge2bit)

        choices = [list(reversed(range(4, bit + 1))) for bit in bits]
        # print('bit choices')
        # edge2choices = complete_dict(choices, topology.edge2cond)
        # print_edge_dict(graph, edge2choices)
        return choices

    def infer_scale_shape(self):
        """For per channel scales"""
        print(self.graph)
        # {edge: (iscale_shape, oscale_shape)}
        edge2idx = self.edge2idx()
        in_scale_shape = [()] * len(edge2idx)
        out_scale_shape = [()] * len(edge2idx)
        qconfig = current_qconfig()
        use_channel_quantized = qconfig.use_channel_quantize
        if not use_channel_quantized:
            return list(zip(in_scale_shape, out_scale_shape))

        node2shape = OrderedDict()
        def fvisit(node):
            if isinstance(node, (relay.Constant, relay.Var, relay.Call)):
                node2shape[node] = node.checked_type.shape
        relay.analysis.post_order_visit(self.graph, fvisit)

        for (src, node), idx in edge2idx.items():
            if isinstance(src, relay.Constant) and node.op.name in qconfig.per_channel_ops():

                if not self.is_quantized_node(node):
                    out_scale_shape[idx] = ()
                    out_edges = self.node2edges()[node]
                    for oedge in out_edges:
                        assert oedge[0] == node
                        in_scale_shape[edge2idx[oedge]] = ()
                else:
                    # Find the output scale size
                    src_layout = self._node2layout[src]
                    assert src_layout in ('OIHW', 'HWIO'), src_layout
                    axis = src_layout.find('O')
                    out_scale_shape[idx] = (node2shape[src][axis],)

                    # Find the input scale size for fan outs
                    out_edges = self.node2edges()[node]
                    node_layout = self._node2layout[node]
                    assert node_layout in ('NCHW', 'NHWC'), node_layout
                    axis = node_layout.find('C')
                    for oedge in out_edges:
                        assert oedge[0] == node
                        in_scale_shape[edge2idx[oedge]] = (node2shape[node][axis],)

        return list(zip(in_scale_shape, out_scale_shape))

    def _build_node_index(self):
        node2idx = OrderedDict()
        def fvisit_build_index(node):
            if isinstance(node, (relay.Var, relay.Constant, relay.Call)):
                node2idx[node] = fvisit_build_index.idx_cnt 
                fvisit_build_index.idx_cnt += 1
        fvisit_build_index.idx_cnt = 0
        relay.analysis.post_order_visit(self.graph, fvisit_build_index)
        num_nodes = fvisit_build_index.idx_cnt
        return node2idx
    
    def _build_edge_index(self):
        edge2idx = OrderedDict() 
        def fvisit_build_index(node):
            if isinstance(node, relay.Call):
                for edge in list_in_edges(node):
                    edge2idx[edge] = fvisit_build_index.idx_cnt 
                    fvisit_build_index.idx_cnt += 1

        fvisit_build_index.idx_cnt = 0
        relay.analysis.post_order_visit(self.graph, fvisit_build_index)
        num_edges = fvisit_build_index.idx_cnt
        return edge2idx

    def _build_node2edges(self):
        node2edges = defaultdict(list)
        def fvisit_build_index(node):
            if isinstance(node, relay.Call):
                for edge in list_in_edges(node):
                    node2edges[edge[0]].append(edge) 
        relay.analysis.post_order_visit(self.graph, fvisit_build_index)
        return node2edges

    def _build_node2kind(self):
        node2kind = OrderedDict()
        def fvisit(node):
            if isinstance(node, relay.Var):
                node2kind[node] = NodeKind.Input
            elif isinstance(node, relay.Constant):
                node2kind[node] = NodeKind.Weight
            elif isinstance(node, relay.Call):
                node2kind[node] = NodeKind.Activation
        relay.analysis.post_order_visit(self.graph, fvisit)
        return node2kind

    def _build_node2layoutinfo(self):
        node2layout = OrderedDict()
        node2channel_axis = OrderedDict()
        def _find_output_axis(layout):
            assert layout in ("OIHW", "HWIO", "NHWC", "NCHW")
            if 'O' in layout:
                return layout.find('O')
            return layout.find('C')

        def fvisit(node):
            if isinstance(node, relay.Var):
                node2layout[node] = "Undef"
                node2channel_axis[node] = -1
            elif isinstance(node, relay.Constant):
                node2layout[node] = "Undef"
                node2channel_axis[node] = -1
            elif isinstance(node, relay.Call):
                if node.op.name == 'nn.conv2d':
                    # Conv2D
                    data_layout = node.attrs.data_layout
                    kernel_layout = node.attrs.kernel_layout
                    out_layout = node.attrs.out_layout
                    out_layout = data_layout if out_layout == "" else out_layout
                    node2layout[node.args[0]] = data_layout
                    node2layout[node.args[1]] = kernel_layout
                    node2layout[node] = out_layout

                    node2channel_axis[node.args[0]] = _find_output_axis(data_layout)
                    node2channel_axis[node.args[1]] = _find_output_axis(kernel_layout)
                    node2channel_axis[node] = _find_output_axis(out_layout)
                elif node.op.name == 'add':
                    # BiasAdd
                    data_layout = node2layout[node.args[0]]
                    if data_layout != "Undef":
                        node2layout[node.args[1]] = data_layout
                        node2layout[node] = data_layout
                        node2channel_axis[node.args[1]] = _find_output_axis(data_layout)
                        node2channel_axis[node] = _find_output_axis(data_layout)
                    else:
                        node2layout[node] = "Undef"
                        node2channel_axis[node] = -1
                else:
                    node2layout[node] = "Undef"
                    node2channel_axis[node] = -1
        relay.analysis.post_order_visit(self.graph, fvisit)
        return (node2layout, node2channel_axis)

    def build_node_info(self, alist):
        ret = OrderedDict()
        cnt = 0
        node2idx = self.node2idx()
        for key, nidx in node2idx.items():
            val = None
            if self._node_conds[nidx]:
                val = alist[cnt]
                cnt += 1
            ret[key] = val
        assert cnt == len(alist)
        return ret
    
    def build_edge_info(self, alist):
        ret = OrderedDict()
        cnt = 0
        edge2idx = self.edge2idx()
        for key, eidx in edge2idx.items():
            val = None
            if self._edge_conds[eidx]:
                val = alist[cnt]
                cnt += 1
            ret[key] = val
        assert cnt == len(alist)
        return ret

    def print_node_info(self, node2info):
        node2idx = self.node2idx(self.graph)
        def fvisit_print(node):
            if isinstance(node, (relay.Var, relay.Constant, relay.Call)):
                print('{}: {}'.format(node_str(node, node2idx), node2info[node]))
        relay.analysis.post_order_visit(graph, fvisit_print)
    
    
    def print_edge_info(self, edge2info):
        node2idx = self.node2idx()
        node2edges = self.node2edges()
        def fvisit_print(node):
            if isinstance(node, relay.Call):
                oedges = node2edges[node]
                out_infos = [edge2info[e] for e in oedges]
                print('--------')
                print('{}: {}'.format(node_str(node, node2idx), out_infos))
                for edge in list_in_edges(node):
                    info = edge2info[edge]
                    print('  {} : {}'.format(edge_str(edge, node2idx), info))
        relay.analysis.post_order_visit(self.graph, fvisit_print)


def analyze_topology(graph, hardware):
    topology = Topology(graph)
    topology.analyze(hardware)
    return topology
