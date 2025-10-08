import numpy as np
from gpal_nn.tasks.driving_bev_sta.datasets.centerline_connector import are_lines_connected, find_and_remove_cycle, has_cycle
import math

def dict2list(lane_markings):
    lane_markings_list = []
    for i in range(len(lane_markings['id'])):
        cur_lane_marking = {
            'id': lane_markings['id'][i],
            'points': lane_markings['points'][i],
            'class': lane_markings['classes'][i],
            'connect_forward_id': [],
            'shape_type': lane_markings['shape_type'][i],
            'color_type': lane_markings['color_type'][i],
            'stop_type': lane_markings['stop_type'][i],
        }
        lane_markings_list.append(cur_lane_marking)
    return lane_markings_list

def list2dict(lane_markings_list):
    lane_marking_dict = {
        "id": [],
        "points": [],
        "classes": [],
        "shape_type": [],
        "color_type": [],
        "stop_type": [],
    }
    for lane_marking in lane_markings_list:
        lane_marking_dict["id"].append(lane_marking["id"])
        lane_marking_dict["points"].append(lane_marking["points"])
        lane_marking_dict["classes"].append(lane_marking["class"])
        lane_marking_dict["shape_type"].append(lane_marking["shape_type"])
        lane_marking_dict["color_type"].append(lane_marking["color_type"])
        lane_marking_dict["stop_type"].append(lane_marking["stop_type"])
    return lane_marking_dict


def is_split_merge_group(cur_id, forward_ids, cross_points=None):
    if cross_points is None:
        return True
    for cross_point in cross_points:
        if cur_id in cross_point and np.all([forward_id in cross_point for forward_id in forward_ids]):
            return True
    return False

def should_connect_lane_markings(lane_marking1, lane_marking2):
    if not are_lines_connected(lane_marking1['points'], lane_marking2['points'], 0.02):
        return False
    if lane_marking1['class'] != lane_marking2['class']:
        return False
    if lane_marking1['shape_type'] != lane_marking2['shape_type']:
        return False
    if lane_marking1['color_type'] != lane_marking2['color_type']:
        return False
    if lane_marking1['stop_type'] != lane_marking2['stop_type']:
        return False
    return True

def connect_lane_markings(lane_markings, cross_points=None):
    lane_markings_list_origin = dict2list(lane_markings)
    lane_markings_list = []
    for lane_marking in lane_markings_list_origin:
        if len(lane_marking['points']) < 2:
            continue
        length = np.sum(np.linalg.norm(np.diff(lane_marking['points'], axis=0), axis=-1)) 
        if length < 1.0:
            continue
        lane_markings_list.append(lane_marking)
    lane_markings_list_new = []
    id2lane_marking = {lane_marking['id']: lane_marking for lane_marking in lane_markings_list}
    for idx1, lane_marking1 in enumerate(lane_markings_list):
        lane_markings_list[idx1]['connect_forward_id'] = []
        for idx2, lane_marking2 in enumerate(lane_markings_list):
            if lane_marking1['id'] == lane_marking2['id']:
                continue
            if should_connect_lane_markings(lane_marking1, lane_marking2) :
                lane_markings_list[idx1]['connect_forward_id'].append(lane_marking2['id'])

    for idx1, lane_marking in enumerate(lane_markings_list):
        if len(lane_marking['connect_forward_id']) > 1:
            is_split_merge_group_flag = is_split_merge_group(lane_marking['id'], lane_marking['connect_forward_id'], cross_points)
            if(is_split_merge_group_flag):
                lane_markings_list[idx1]['connect_forward_id'] = []
            # else:
            #     print("has_two_forward: ", json_file, lane_marking['id'], lane_marking['connect_forward_id'], cross_points)


    backward_map = {lane_marking['id']: [] for lane_marking in lane_markings_list}
    for lane_marking in lane_markings_list:
        for forward_id in lane_marking['connect_forward_id']:
            backward_map[forward_id].append(lane_marking['id'])
    id2idx = {lane_marking['id']: idx for idx, lane_marking in enumerate(lane_markings_list)}
    for idx1, lane_marking in enumerate(lane_markings_list):
        cur_id = lane_marking['id']
        backward_ids = backward_map[cur_id]
        if len(backward_ids) > 1:
            is_split_merge_group_flag = is_split_merge_group(cur_id, backward_ids, cross_points)
            if(is_split_merge_group_flag):
                for backward_id in backward_ids:
                    lane_markings_list[id2idx[backward_id]]['connect_forward_id'].remove(cur_id)
            # else:
            #     print("has_two_backward: ", json_file, cur_id, backward_ids, cross_points)


    graph = {lane_marking['id']: lane_marking["connect_forward_id"] for lane_marking in lane_markings_list}
    has_cycle_flag = has_cycle(graph)
    while has_cycle_flag:
        # print("has_cycle", json_file)
        graph, has_cycle_flag = find_and_remove_cycle(graph)
    assert not has_cycle(graph)
    for idx, lane_marking in enumerate(lane_markings_list):
        lane_markings_list[idx]['connect_forward_id'] = graph[lane_marking['id']]

    in_degree = {lane_marking['id']: 0 for lane_marking in lane_markings_list}
    for lane_marking in lane_markings_list:
        for forward_id in lane_marking['connect_forward_id']:
            in_degree[forward_id] += 1
    root_ids = [lane_marking['id'] for lane_marking in lane_markings_list if in_degree[lane_marking['id']] == 0]

    for root_id in root_ids:
        cur_lane_marking = {
            'id': root_id,
            'points': [id2lane_marking[root_id]['points']],
            'class': id2lane_marking[root_id]['class'],
            'shape_type': id2lane_marking[root_id]['shape_type'],
            'color_type': id2lane_marking[root_id]['color_type'],
            'stop_type': id2lane_marking[root_id]['stop_type'],
        }
        cur_id = root_id
        while(len(id2lane_marking[cur_id]['connect_forward_id']) > 0):
            assert len(id2lane_marking[cur_id]['connect_forward_id']) == 1
            # print(f"connect lane_marking {cur_id} 到 {id2lane_marking[cur_id]['connect_forward_id'][0]}")
            forward_id = id2lane_marking[cur_id]['connect_forward_id'][0]
            cur_lane_marking['points'].append(id2lane_marking[forward_id]['points'])
            cur_id = forward_id
        cur_lane_marking['points'] = np.concatenate(cur_lane_marking['points'], axis=0)
        lane_markings_list_new.append(cur_lane_marking)
    lane_markings = list2dict(lane_markings_list_new)

    return lane_markings