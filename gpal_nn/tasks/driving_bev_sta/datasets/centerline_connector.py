import numpy as np
# import matplotlib.pyplot as plt
from collections import defaultdict
# import json
import math
# import os
from tqdm import tqdm

def get_behavior_type(old_type, new_type):
    if old_type == 'normal':
        return new_type
    else:
        return old_type
    
def get_centerline_class(old_class, new_class):
    if old_class == 'normal_lane':
        return new_class
    else:
        return old_class

def get_centerline_dict(centerline_list):
    centerline_dict = {
        "id": [],
        "points": [],
        "classes": [],
        "behavior_type": [],
        "connect_forward_id": [],
        "is_split_merge": [],
        "keypoint": [],
    }
    for centerline in centerline_list:
        centerline_dict["id"].append(centerline["id"])
        centerline_dict["points"].append(centerline["points"])
        centerline_dict["classes"].append(centerline["class"])
        centerline_dict["behavior_type"].append(centerline["behavior_type"])
        centerline_dict["connect_forward_id"].append(centerline["connect_forward_id"])
        centerline_dict["is_split_merge"].append(centerline["is_split_merge"])
        centerline_dict["keypoint"].append(centerline["keypoint"])
    return centerline_dict

def find_and_remove_cycle(graph):
    """检测并删除图中的环
    Args:
        graph: 邻接列表表示的有向图
    Returns:
        tuple: (修改后的图, 是否删除了环)
    """
    visited = set()
    recursion_stack = []  # 使用列表记录递归路径
    cycle_edges = None     # 存储环中的一条边

    def dfs(node):
        nonlocal cycle_edges
        if node not in visited:
            visited.add(node)
            recursion_stack.append(node)

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in recursion_stack:
                    # 找到环：从neighbor到当前node的路径
                    cycle_index = recursion_stack.index(neighbor)
                    cycle_path = recursion_stack[cycle_index:] + [neighbor]
                    cycle_edge_count = len(cycle_path) - 1
                    if cycle_edge_count == 2:
                        cycle_edges = [(cycle_path[-2], cycle_path[-1]), (cycle_path[-1], cycle_path[-2])]
                    else:
                        # print("has cycle, and cycle_edge_count != 2")
                        cycle_edges = [(cycle_path[-2], cycle_path[-1])]
                    return True

            recursion_stack.pop()
        return False

    # 检测环并获取环中的一条边
    for node in graph:
        if node not in visited:
            if dfs(node):
                # 删除环中的一条边
                if cycle_edges:
                    for cycle_edge in cycle_edges:
                        from_node, to_node = cycle_edge
                        graph[from_node].remove(to_node)
                        # print(f"检测到环，已删除边: {cycle_edge}")
                    return graph, True
    return graph, False

def has_cycle(graph):
    """判断有向图是否存在环
    Args:
        graph: 邻接列表表示的有向图，格式为 {节点ID: [相邻节点ID列表]}
    Returns:
        bool: 存在环返回True，否则返回False
    """
    visited = set()          # 记录已访问的节点
    recursion_stack = set()  # 记录当前递归路径中的节点

    def dfs(node):
        if node not in visited:
            visited.add(node)
            recursion_stack.add(node)

            # 遍历所有相邻节点
            for neighbor in graph.get(node, []):
                # 若邻居未访问且递归检测到环，或邻居已在递归栈中（发现回溯边）
                if neighbor not in visited and dfs(neighbor):
                    return True
                elif neighbor in recursion_stack:
                    return True

            # 回溯时移出递归栈
            recursion_stack.remove(node)
        return False

    # 对所有未访问节点执行DFS
    for node in graph:
        if dfs(node):
            return True
    return False

def are_lines_connected(line1, line2, epsilon=0.2):
    """
    判断两根散点线是否首尾相连
    
    参数:
        line1: 第一根线的散点列表，每个元素为(x,y)元组或列表
        line2: 第二根线的散点列表
        epsilon: 距离阈值，小于此值认为两点重合
    
    返回:
        bool: 是否首尾相连
    """
    end1 = line1[-1]
    start2 = line2[0]
    return math.hypot(end1[0]-start2[0], end1[1]-start2[1]) < epsilon

def calculate_distance(point1, point2):
    """计算两点之间的欧氏距离"""
    return math.hypot(point2[0] - point1[0], point2[1] - point1[1])

def build_centerline_graph(centerline_list):
    """构建centerline连接关系图"""
    # 创建ID到centerline的映射

    centerline_list_new = []
    for centerline in centerline_list:
        if len(centerline['points']) < 2:
            continue
        length = np.sum(np.linalg.norm(np.diff(centerline['points'], axis=0), axis=-1)) 
        if length < 1.0:
            continue
        centerline_list_new.append(centerline)
    centerline_list = centerline_list_new
    id_to_centerline = {p['id']: p for p in centerline_list}
    
    # 构建有向图 adjacency list
    graph = defaultdict(list)

    for centerline_idx, p in enumerate(centerline_list):
        for q in centerline_list:
            if p['id'] == q['id']:
                continue
            if are_lines_connected(p['points'], q['points']) and q['id'] not in p['connect_forward_id']:
                # print(f"connect {p['id']} 到 {q['id']}")
                centerline_list[centerline_idx]['connect_forward_id'].append(q['id'])  

    # 构建图并确定实际起始节点
    for p in centerline_list:
        current_id = p['id']
        graph[current_id] = []
        for forward_id in p['connect_forward_id']:
            if forward_id not in id_to_centerline:
                continue
            q = id_to_centerline[forward_id]

            # 获取四个端点
            p_head = p['points'][0]
            p_tail = p['points'][-1]
            q_head = q['points'][0]
            q_tail = q['points'][-1]
            
            # 计算四种首尾组合的距离
            d1 = calculate_distance(p_head, q_head)  # p首 -> q首
            d2 = calculate_distance(p_head, q_tail)  # p首 -> q尾
            d3 = calculate_distance(p_tail, q_head)  # p尾 -> q首
            d4 = calculate_distance(p_tail, q_tail)  # p尾 -> q尾
            
            # 找出最小距离
            min_distance = min(d1, d2, d3, d4)
            # 检查最小距离是否是p尾到q首
            if abs(min_distance - d3) < 1e-6 and min_distance < 1.0:
                graph[current_id].append(forward_id)

    return graph, id_to_centerline


def merge_connected_centerlines(centerlines_list):
    """合并所有连接的centerline"""
    graph, id_to_centerline = build_centerline_graph(centerlines_list)
    has_cycle_flag = has_cycle(graph)
    while has_cycle_flag:
        graph, has_cycle_flag = find_and_remove_cycle(graph)
    assert not has_cycle(graph)
    merged_centerlines = []
    all_ids = set(graph.keys())
    in_degree = {pid: 0 for pid in all_ids}
    out_degree = {pid: 0 for pid in all_ids}
    new_id = 1
    
    # 计算入度以识别根节点
    for pid in graph:
        for neighbor in graph[pid]:
            in_degree[neighbor] += 1
    for pid in graph:
        out_degree[pid] = len(graph[pid])
    # 根节点(入度为0)和未处理节点
    root_nodes = [pid for pid in all_ids if in_degree[pid] == 0]
    
    # 处理根节点
    for root in root_nodes:
        queue = [(root, [root], id_to_centerline[root]['points'].tolist(), id_to_centerline[root]['class'], id_to_centerline[root]['behavior_type'])]
        while queue:
            current_id, path, points, centerline_class, behavior_type = queue.pop(0)
            
            neighbors = graph.get(current_id, [])
            if len(neighbors) == 0:
                # 叶子节点，添加完整路径
                merged_centerlines.append({
                    'id': new_id,
                    'path': path,
                    'points': points,
                    'class': centerline_class,
                    'behavior_type': behavior_type,
                    'connect_forward_id': [],
                    'is_split_merge': False,
                    'keypoint': np.zeros(3),
                })
                new_id += 1
                continue
            
            # 处理分叉 - 为每个邻居创建路径副本
            for i, neighbor in enumerate(reversed(neighbors)):
                
                # 复制当前路径和点列表
                new_path = path.copy()
                new_path.append(neighbor)
                
                new_points = points.copy()
                new_points.extend(id_to_centerline[neighbor]['points'].tolist())
                
                queue.append((neighbor, new_path, new_points, 
                              get_centerline_class(centerline_class, id_to_centerline[neighbor]['class']), \
                              get_behavior_type(behavior_type, id_to_centerline[neighbor]['behavior_type'])))

    for merged_idx in range(len(merged_centerlines)):
        for centerline_idx in range(len(merged_centerlines[merged_idx]['path'])):
            if in_degree[merged_centerlines[merged_idx]['path'][centerline_idx]] > 1:
                merged_centerlines[merged_idx]['is_split_merge'] = True
                merged_centerlines[merged_idx]['keypoint'] = \
                np.array(id_to_centerline[merged_centerlines[merged_idx]['path'][centerline_idx]]['points'][0])
                break
            if out_degree[merged_centerlines[merged_idx]['path'][centerline_idx]] > 1:
                merged_centerlines[merged_idx]['is_split_merge'] = True
                merged_centerlines[merged_idx]['keypoint'] = \
                    np.array(id_to_centerline[merged_centerlines[merged_idx]['path'][centerline_idx]]['points'][-1])
                break
    return merged_centerlines

