import numpy as np

# --------------------------- line -----------------------------------
map_classes_line = ['normal', 'fishbone', 'stop_line', 'cross_guide_line', 'wait_line', 'bike_cross_line',
                    'ignore', 'others']
shape_type = ['single_solid', 'single_dashed', 'double_solid', 'double_dashed', 'double_left_solid',
              'double_right_solid', 'colored_three_line', 'thick_dashed', 'reversible_line', 'variable_line',
              'point_line', 'others']
color_type = ['white', 'yellow', 'blue', 'orange', 'red', 'others']
stop_type = ['normal', 'giveaway', 'slowdown', 'virtual', 'others']

# --------------------------- edge --------------------------------------------------------------
map_classes_edge = ['plain_roadside', 'roadside', 'cone', 'waterhorse', 'fence', 'others']

# --------------------------- centerline --------------------------------------------------------------
map_class_centerline = ["centerline"]
# connect_forward_id = []
# connect_backward_id = []

# --------------------------- polygon ---------------------------------------------------------
map_classes_polygon = ['arrow', 'crosswalk', 'no_parking_zone', 'left_waiting_zone', 'straight_waiting_zone',
                       'right_waiting_zone', 'slow_down_zone', 'others']
# arrow_type = ['直行箭头', '直行或向左转弯', '直行或向右转弯', '直行或调头', '左转',
#                 '左转或调头', '左转或向左合流', '右转', '右弯或向右合流','左右转弯',
#                 '调头', '禁止左转标记', '禁止右转标记', '禁止掉头标记',
#                 '直行或左转或右转', '直行或调头或左转', '右转或调头', '人行横道预告标识', '让路线', '非机动车道指向', 'others']
arrow_type = ['straight', 'straight_turnleft', 'straight_turnright', 'straight_uturn', 'turnleft',
              'turnleft_uturn', 'turnleft_mergeleft', 'turnright', 'turnright_mergeright', 'turnleft_turnright',
              'uturn', 'prohibited_turnleft', 'prohibited_turnright', 'prohibited_uturn',
              'straight_turnleft_turnright', 'straight_uturn_turnleft', 'turnright_uturn', 'pedestrian_warning',
              'giveway', 'bike_direction', 'others']

# --------------------------- parking_slot --------------------------------------------------------------
map_class_parking = ['vertical_parking_slot', 'horizontal_parking_slot', 'inclined_parking_slot',
                     'big_vertical_parking_slot', 'big_horizontal_parking_slot', 'big_inclined_parking_slot']
occupied_type = ['occupied', 'not_occupied']

# --------------------------- point -------------------------------------------------------------
map_classes_point = ['split_point', 'merge_point']