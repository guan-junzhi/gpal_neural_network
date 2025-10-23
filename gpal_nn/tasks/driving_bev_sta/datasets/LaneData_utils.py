import numpy as np

main_class_type_map = {
    "lane_marking": 0,
    "edge": 1,
    "centerline": 2,
    "polygon": 3,
    "arrow": 4,
    "parking_slot": 5,
}

main_class_name_map = {index: name for name, index in main_class_type_map.items()}

# --------------------------- lane_marking -----------------------------------
lane_marking_type_map = {
    "normal": 0,
    "guide_line": 1,
    "fishbone": 2,
    "stop_line": 3,
    "cross_guide_line": 4,
    'cross_guide_ line': 4,
    "bike_cross_line": 5,
    "wait_line": 6,    # 兼容老标注规范
    "waiting_line": 6,
    "ignore": -1,
    "others": -1,
}

shape_type_map = {
    "single_solid": 0,
    "single_dashed": 1,
    "double_soild": 5,  # 兼容标注规范拼写错误
    "double_solid": 5,
    "double_dashed": 6,
    "double_left_soild": 2,  # 兼容标注规范拼写错误
    "double_left_solid": 2,
    "double_right_soild": 3,  # 兼容标注规范拼写错误
    "double_right_solid": 3,
    "thick_dashed": 4,
    "wide_solid": 7,
    "colored_three_line": 8,
    "reversible_line": 9,
    "variable_line": 10,
    "point_line": 11,
    "others": -1,
}

color_type_map = {
    "white": 0,
    "yellow": 1,
    "blue": -1,
    "orange": -1,
    "red": -1,
    "others": -1,
}

stop_type_map = {
    "normal": 0,
    "giveaway": 0,
    "slowdown": 0,
    "virtual": 0,
    "others": -1,
}

# --------------------------- edge --------------------------------------------------------------
edge_type_map = {
    "plain_roadside": 0,
    "roadside": 0,
    "fence": 0,
    "stone_pier": 0,
    "seprate_stone_pier": 0,
    "wall": 0,
    "cone": 0,
    "waterhorse": 0,
    "pillar": 0,
    "bumper_barrel": 0,
    "others": -1,
}

# --------------------------- centerline --------------------------------------------------------------
centerline_type_map = {
    "normal_lane": 0,
    "emergency_lane": 1,
    "non_motorized_lane": 0,
    "turn_waiting_lane": 0,
    "intersection_lane": 0,
    "intersection_virtual_lane": 0,
    "bus_lane": 0,
    "bus_island_lane": 0,
    "parking_island_lane": 0,
    "rechannel_lane": 0,
    "temporary_lane": 0,
    "toll_station_lane": 0,
    "variable_lane": 0,
    "reversible_lane": 0,
    "ramp_lane": 0,
    "others": -1,
}

behavior_type_map = {
    "normal": 0,
    "side": 1,
    "others": -1,
}
# --------------------------- polygon ---------------------------------------------------------
polygon_type_map = {
    "arrow": 0,
    "speed_bump": 0,
    "deceleration_horizontal": 0,
    "crosswalk": 0,
    "no_parking_zone": 0,
    "diversion_tape": 0,
    "left_waiting_zone": 0,
    "straight_waiting_zone": 0,
    "right_waiting_zone": 0,
    "other_waiting_zone": 0,
    "slow_down_zone": 0,
    "danger_zone": 0,
    "vertical_parking_slot": 0,
    "horizontal_parking_slot": 0,
    "inclined_parking_slot": 0,
    "big_vertical_parking_slot": 0,
    "big_horizontal_parking_slot": 0,
    "big_inclined_parking_slot": 0,
    "others": -1,
    "other": -1,
}

arrow_type_map = {
    "straight": 0,
    "straight_turnleft": 0,
    "straight_turnright": 0,
    "straight_uturn": 0,
    "turnleft": 0,
    "turnleft_uturn": 0,
    "turnleft_mergeleft": 0,
    "nleft_mergeleft": 0,
    "turnright": 0,
    "turnright_mergeright": 0,
    "turnleft_turnright": 0,
    "uturn": 0,
    "prohibited_turnleft": 0,
    "prohibited_turnright": 0,
    "prohibited_uturn": 0,
    "straight_turnleft_turnright": 0,
    "straight_uturn_turnleft": 0,
    "turnright_uturn": 0,
    "pedestrian_warning": 0,
    "intersection_center_circle": 0,
    "giveway": 0,
    "bike_direction": 0,
    "others": -1,
    None:-1,
    "": -1,
}

# --------------------------- parking_slot --------------------------------------------------------------
parking_slot_type_map = {
    "vertical_parking_slot": 0,
    "horizontal_parking_slot": 0,
    "inclined_parking_slot": 0,
    "big_vertical_parking_slot": 0,
    "big_horizontal_parking_slot": 0,
    "big_inclined_parking_slot": 0,
    "others": -1,
}
occupied_type_map = {
    "occupied": 0,
    "not_occupied": 1,
    "others": -1,
}

# --------------------------- point -------------------------------------------------------------
point_type_map = {
    "split_point": 0,
    "merge_point": 1,
    "type_change": 2,
    "others": -1,
}
