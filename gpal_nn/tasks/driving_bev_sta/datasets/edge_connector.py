import numpy as np
from gpal_nn.tasks.driving_bev_sta.datasets.centerline_connector import are_lines_connected, find_and_remove_cycle, has_cycle
import math

def dict2list(edges):
    edges_list = []
    for i in range(len(edges['id'])):
        cur_edge = {
            'id': edges['id'][i],
            'points': edges['points'][i],
            'class': edges['classes'][i],
            'connect_forward_id': [],
        }
        edges_list.append(cur_edge)
    return edges_list

def list2dict(edges_list):
    edge_dict = {
        "id": [],
        "points": [],
        "classes": [],
    }
    for edge in edges_list:
        edge_dict["id"].append(edge["id"])
        edge_dict["points"].append(edge["points"])
        edge_dict["classes"].append(edge["class"])
    return edge_dict

def densify_instance(input, num_points=20):
    diff = np.diff(input, axis=0)
    seg_dist = np.linalg.norm(diff, axis=1)
    # 累积距离作为插值x轴
    cum_dist = np.concatenate([[0], np.cumsum(seg_dist)])
    total_length = cum_dist[-1]
    # 生成插值点距离
    interp_dist = np.linspace(0, total_length, num_points)
    # 对x,y,z分别进行插值
    x_interp = np.interp(interp_dist, cum_dist, input[:, 0])
    y_interp = np.interp(interp_dist, cum_dist, input[:, 1])
    # 合并结果
    output = np.column_stack([x_interp, y_interp])

    return output

def direction_similar(edge1, edge2):
    edge1_pts = densify_instance(edge1['points'][:,:2])
    edge1_pt1 = edge1_pts[-2]
    edge1_pt2 = edge1_pts[-1]

    edge2_pts = densify_instance(edge2['points'][:,:2])
    edge2_pt1 = edge2_pts[0]
    edge2_pt2 = edge2_pts[1]

    vec1 = edge1_pt2 - edge1_pt1
    vec2 = edge2_pt2 - edge2_pt1
    
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    # 避免除以零
    if norm_vec1 == 0 or norm_vec2 == 0:
        return False
    
    dot_product1 = np.dot(vec1, vec2)
    cos_theta1 = np.clip(dot_product1 / (norm_vec1 * norm_vec2), -1.0, 1.0)
    angle_deg1 = np.degrees(abs(np.arccos(cos_theta1)))
    if angle_deg1 > 120.0:
        return False
    return True

def calculate_overlap_length(edge1, edge2):

    edge1_min_x = np.min(edge1['points'][:, 0])
    edge1_max_x = np.max(edge1['points'][:, 0])
    edge2_min_x = np.min(edge2['points'][:, 0])
    edge2_max_x = np.max(edge2['points'][:, 0])
    overlap_start = max(edge1_min_x, edge2_min_x)
    overlap_end = min(edge1_max_x, edge2_max_x)
    
    if overlap_start < overlap_end:
        return overlap_end - overlap_start
    else:
        return 0
    
def should_connect(edge1, edge2):
    if edge1['class'] != edge2['class']:
        return False
    
    if not are_lines_connected(edge1['points'], edge2['points'], 0.2):
        return False
    
    if calculate_overlap_length(edge1, edge2) > 10:
        return False

    if not direction_similar(edge1, edge2):
        return False

    return True

def connect_edges(edges):
    edges_list_origin = dict2list(edges)
    edges_list = []
    for edge in edges_list_origin:
        if len(edge['points']) < 2:
            continue
        length = np.sum(np.linalg.norm(np.diff(edge['points'], axis=0), axis=-1)) 
        if length < 1.0:
            continue
        edges_list.append(edge)

    edges_list_new = []
    id2edge = {edge['id']: edge for edge in edges_list}
    for idx1, edge1 in enumerate(edges_list):
        for idx2, edge2 in enumerate(edges_list):
            if edge1['id'] == edge2['id']:
                continue
            if should_connect(edge1, edge2):
                edges_list[idx1]['connect_forward_id'].append(edge2['id'])

    for idx1, edge in enumerate(edges_list):
        if len(edge['connect_forward_id']) > 1:
            dist_list = []
            for forward_id in edge['connect_forward_id']:
                dist_list.append(math.hypot(edge['points'][-1][0]-id2edge[forward_id]['points'][0][0], edge['points'][-1][1]-id2edge[forward_id]['points'][0][1]))
            min_idx = np.argmin(dist_list)
            edges_list[idx1]['connect_forward_id'] = [edge['connect_forward_id'][min_idx]]

    backward_map = {edge['id']: [] for edge in edges_list}
    for edge in edges_list:
        for forward_id in edge['connect_forward_id']:
            backward_map[forward_id].append(edge['id'])
    id2idx = {edge['id']: idx for idx, edge in enumerate(edges_list)}
    for idx1, edge in enumerate(edges_list):
        cur_id = edge['id']
        backward_ids = backward_map[cur_id]
        dist_list = []
        if len(backward_ids) > 1:
            for backward_id in backward_ids:
                dist_list.append(math.hypot(id2edge[backward_id]['points'][-1][0]-edge['points'][0][0], id2edge[backward_id]['points'][-1][1]-edge['points'][0][1]))
            min_idx = np.argmin(dist_list)
            for i in range(len(backward_ids)):
                if i != min_idx:
                    edges_list[id2idx[backward_ids[i]]]['connect_forward_id'].remove(cur_id)

    graph = {edge['id']: edge["connect_forward_id"] for edge in edges_list}
    has_cycle_flag = has_cycle(graph)
    while has_cycle_flag:
        # print("has_cycle", json_file)
        graph, has_cycle_flag = find_and_remove_cycle(graph)
    assert not has_cycle(graph)
    for idx, edge in enumerate(edges_list):
        edges_list[idx]['connect_forward_id'] = graph[edge['id']]

    in_degree = {edge['id']: 0 for edge in edges_list}
    for edge in edges_list:
        for forward_id in edge['connect_forward_id']:
            in_degree[forward_id] += 1
    root_ids = [edge['id'] for edge in edges_list if in_degree[edge['id']] == 0]

    for root_id in root_ids:
        cur_edge = {
            'id': root_id,
            'points': [id2edge[root_id]['points']],
            'class': id2edge[root_id]['class'],
        }
        cur_id = root_id
        while(len(id2edge[cur_id]['connect_forward_id']) > 0):
            assert len(id2edge[cur_id]['connect_forward_id']) == 1
            forward_id = id2edge[cur_id]['connect_forward_id'][0]
            cur_edge['points'].append(id2edge[forward_id]['points'])
            cur_id = forward_id
        cur_edge['points'] = np.concatenate(cur_edge['points'], axis=0)
        edges_list_new.append(cur_edge)
    edges = list2dict(edges_list_new)

    return edges