import csv
from operator import itemgetter

import scipy.io as sio
import numpy as np
import pandas as pd
from fault_diagnosis.utility.MyException import MyException
# import networkx as nx
# from networkx.algorithms import community
from igraph import *


# from fault_diagnosis.utility.pylouvain import PyLouvain


class ConfusionGraphMaker(object):

    def __init__(self, predict_data_path='./data/test/predict_score.mat', top_n=5):
        """
            初始化制作器
            参数：
                :param predict_data_path:   分类预测结果数据文件路径
                :param vertex:              混淆图顶点集合（用各个样本类别号代表各个顶点）
                :param top_n:               超参数：只取多分类得分的前 n 个分类结果进行计算
                :return:                    混淆图的顶点集和边的权重矩阵
        """
        _data = sio.loadmat(predict_data_path)
        self._predict = _data['predict']
        self._label = _data['label_test'][0]
        self._faultid = _data['faultid_test'][0]
        self._fault_ids = _data['fault_ids'][0]
        self._test_sample_count = self._predict.shape[0]
        self._number_categories = self._predict.shape[1]
        # self._vertexes = range(0, self._number_categories)
        # self._edges = None  # 用字典来表示边：(v_i, v_j):w_ij
        self._top_n = top_n

    def create_confusion_graph_ijcai(self):
        edges = {}
        nodes = {}
        for i in range(0, self._test_sample_count):
            # 获取当前测试样本的类标签
            i_label = self._label[i]
            # 如果该样本的类标签不在结点字典中存在，则添加一个新结点到图顶点集合中，同时记录该类标签对应到原始故障类别号
            if i_label not in nodes:
                nodes[i_label] = self._faultid[i]
            top_c, top_s = self.__get_top_n_info(self._predict[i], self._top_n)
            # 对预测概率值进行归一化处理
            top_s_normal = self.__score_normalization(top_s)

            for j in range(0, self._top_n):

                if top_c[j] != i_label:
                    # 当预测得分大于0时，添加边
                    if top_s_normal[j] > 0:
                        if (i_label, top_c[j]) in edges:
                            # 如果边已经在字典中，则更新边到权重
                            new_weight = edges[(i_label, top_c[j])] + top_s_normal[j]
                            edges[(i_label, top_c[j])] = new_weight
                        elif (top_c[j], i_label) in edges:
                            new_weight = edges[top_c[j], i_label] + top_s_normal[j]
                            edges[(top_c[j], i_label)] = new_weight
                            # print(i_label)
                            # print(top_c[j])
                            # print("----------")
                        else:
                            # 如果边不存在，则添加一个边到字典中
                            edges[(i_label, top_c[j])] = top_s_normal[j]
                            # print(i_label)
                            # print(top_c[j])
                            # print("----------")
        return nodes, edges

    def get_community(self, nodes, edges, threshold=None):
        """调用igraph的功能进行社区发现"""
        g = Graph()

        # 默认使用中位数作为边的权重过滤阈值
        if threshold is None:
            threshold = self.get_median(edges, 1)

        # 构造边表
        _edges = []
        for k, v in edges.items():
            if v > threshold:
                # 使用原始故障类别号作为结点的name，方便观察结果
                _edges.append((str(nodes[k[0]]), str(nodes[k[1]]), v))

        g = Graph.TupleList(edges=_edges, directed=False, edge_attrs=None, weights=True)

        # for i in nodes.keys():
        #     g.add_vertices(str(nodes[i]))
        # g.vs['label'] = nodes.values()

        c = g.community_multilevel(weights='weight', return_levels=False)
        return g, c

    def print_communities(self, graph: Graph, communities, filepath=None):
        str = ''
        str_line = '\t%s' % ('-'*40)
        i = 0
        for c in communities:
            i += 1
            msg_comm = '\ncommunity[%d]: \t%s\n%s\n' % (i, graph.vs[c]['name'], str_line)
            str += msg_comm
            print(msg_comm)
            g = graph.subgraph(c)
            for e in g.es:
                # print('edge[%s,%s]=%.5f\n' % (g.vs[e.source]['name'], g.vs[e.target]['name'], e['weight']))
                msg_edge = '\tedge%s = %.5f\n' % (g.vs[e.tuple]['name'], e['weight'])
                str += msg_edge
                print(msg_edge)
            str += str_line
            print(str_line)
        # 保存结果到文本文件
        if filepath is not None:
            with open(filepath, 'w') as f:
                f.write(str)
                f.close()
        return

    def get_median(self, edges, threshold=None):
        """寻找中位数"""
        w = []
        if threshold is None:
            threshold = 1
        # 去掉小于指定阈值的元素
        for v in edges.values():
            if v > threshold:
                w.append(v)
        # 排序
        w = [k for k in sorted(w)]
        m = w[len(w) // 2]
        return m

    def __get_top_n_info(self, t, top_n):
        """获取top N个预测类别概率
            参数：
            :param: 一个样本被分到每一个类别的概率值列表
            :top_n: 取最大的N个值
            返回值：
            :returns: (topN元素下标列表, topN元素对值列表)
        """
        # 参数是list，转成array
        _t_array = np.array(t)
        # 返回最大top_n个值对应的下标，下标值就是对应的类别号
        top_categories = np.argpartition(_t_array, -top_n)[-top_n:]
        # 返回最大top_n个值，注意：结果集元素没有被排序
        top_scores = t[top_categories]
        return top_categories.tolist(), top_scores.tolist()

    def __score_normalization(self, _top_s):
        sum_top_n_scores = sum(_top_s)
        top_s_normal = [item / sum_top_n_scores for item in _top_s]
        return top_s_normal

    @staticmethod
    def create_gephi_csv(nodes_path='', edges_path='', nodes={}, edges={}):
        """
        # 功能：将一字典写入到csv文件中
            # 输入：文件名称，数据字典
            :param fileName:
            :param dataDict:
            :return:
        """
        with open(nodes_path, "w") as nf:
            nf.write('id,label\n')
            [nf.write('{0},{1}\n'.format(key, value)) for key, value in nodes.items()]
            nf.close()
        print("%s has been created successfully." % nodes_path)
        with open(edges_path, "w") as ef:
            ef.write('source,target,type,label,weight\n')
            [ef.write('{0},{1},undirected,{2},{3}\n'.format(key[0], key[1], value, value))
             for key, value in edges.items()]
            ef.close()
        print("%s has been created successfully." % edges_path)

        return
    #
    # def _get_fault_id(self, label):
    #     return self._fault_ids[label]


if __name__ == "__main__":
    cgm = ConfusionGraphMaker(top_n=5)
    nodes, edges = cgm.create_confusion_graph_ijcai()
    g, c = cgm.get_community(nodes, edges)
    cgm.print_communities(g, c)

