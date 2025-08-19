import torch


def DrivingBevStaRemap(checkpoint1_keys, checkpoint2_keys):
    remap_all = {}
    backbone1 = [ele for ele in checkpoint1_keys if "backbone." in ele]
    backbone2 = [ele for ele in checkpoint2_keys if "backbone0." in ele]

    remap_2_to_1 = {ele: ele.replace(
        "model.backbone0.", "backbone.") for ele in backbone2}
    remap_all.update(remap_2_to_1)
    sum([remap_2_to_1[ele] in backbone1 for ele in backbone2])

    print(len(backbone1), len(backbone2), sum(
        [remap_2_to_1[ele] in backbone1 for ele in backbone2]))
    # print(backbone1)
    # print(backbone2)

    neck1 = [ele for ele in checkpoint1_keys if ("neck." in ele)]
    neck2 = [ele for ele in checkpoint2_keys if ("neck0." in ele)]
    remap_2_to_1 = {ele: ele.replace(
        "model.neck0.", "neck.") for ele in neck2}
    remap_all.update(remap_2_to_1)
    print(len(neck1), len(neck2), sum(
        [remap_2_to_1[ele] in neck1 for ele in neck2]))

    vt1 = [ele for ele in checkpoint1_keys if ("vt." in ele)]
    vt2 = [ele for ele in checkpoint2_keys if ("model.transformer." in ele)]
    remap_2_to_1 = {ele: ele.replace(
        "model.transformer.", "vt.") for ele in vt2}
    remap_all.update(remap_2_to_1)
    print(len(vt1), len(vt2), sum(
        [remap_2_to_1[ele] in vt1 for ele in vt2]))

    head1 = [ele for ele in checkpoint1_keys if ("lane_map_head." in ele)]
    head2 = [ele for ele in checkpoint2_keys if (
        "model.DRIVING_BEV_STA.head.head1." in ele)]
    remap_2_to_1 = {ele: ele.replace(
        "model.DRIVING_BEV_STA.head.head1.", "lane_map_head.") for ele in head2}
    remap_all.update(remap_2_to_1)
    print(len(head1), len(head2), sum(
        [remap_2_to_1[ele] in head1 for ele in head2]))

    others1 = [ele for ele in checkpoint1_keys if not (
        ("backbone." in ele) or ("neck." in ele) or ("vt." in ele) or ("lane_map_head." in ele))]
    others2 = [ele for ele in checkpoint2_keys if not (
        ("backbone0." in ele) or ("neck0." in ele) or ("model.transformer." in ele) or ("model.DRIVING_BEV_STA.head.head1." in ele))]

    print(len(others1), len(others2))
    print(others1)
    print(others2)

    return remap_all


def DrivingBevSta():
    ckpt1 = "/data/ai_group/workdirs/multitask_lanenet_group/wujianlong/multitask_lanenet/work_dir/hat_ori_rpy_bev_aug_1deg_0726/model/ep022.pth"
    checkpoint1 = torch.load(ckpt1, map_location="cpu")
    checkpoint1_keys = list(checkpoint1.keys())
    # print(checkpoint1_keys)

    ckpt2 = "workspace/20250728_13_04_38_2epoch_ckpt/checkpoint/epoch=1-step=3500_checkpoint.pth"
    checkpoint2 = torch.load(ckpt2, map_location="cpu")
    # print(checkpoint2['state_dict'].keys())
    checkpoint2_keys = list(checkpoint2['state_dict'].keys())
    print(checkpoint2_keys)

    remap_all = DrivingBevStaRemap(checkpoint1_keys, checkpoint2_keys)
    print(len(checkpoint1_keys), len(checkpoint2_keys), len(remap_all))

    for k in checkpoint2_keys:
        if checkpoint2['state_dict'][k].shape != checkpoint1[remap_all[k]].shape:
            print(k, checkpoint2['state_dict'][k].shape, checkpoint1[remap_all[k]].shape,
                  checkpoint2['state_dict'][k].shape == checkpoint1[remap_all[k]].shape)
        checkpoint2['state_dict'][k] = checkpoint1[remap_all[k]]
    #                    lane_map_head.decoder.layers.0.sa.in_proj_weight
    # model.DRIVING_BEV_STA.head.head1.decoder.layers.0.sa.in_proj_weight
    ckpt2_edit = ckpt2.replace(".pth", "_wangtong.pth")
    print(ckpt2_edit)
    torch.save(checkpoint2, ckpt2_edit)

    bias = checkpoint2['state_dict']["model.DRIVING_BEV_STA.head.head1.input_proj.bias"]
    weight = checkpoint2['state_dict']["model.DRIVING_BEV_STA.head.head1.input_proj.weight"]

    print(bias.min(), bias.max())
    print(weight.min(), weight.max())

    bias = checkpoint1["lane_map_head.input_proj.bias"]
    weight = checkpoint1["lane_map_head.input_proj.weight"]

    print(bias.min(), bias.max())
    print(weight.min(), weight.max())


def ParkingIpmStaRemap(checkpoint1_keys, checkpoint2_keys):
    remap_all = {}
    backbone1 = [ele for ele in checkpoint1_keys if "share_bb." in ele]
    backbone2 = [ele for ele in checkpoint2_keys if "model.backbone0." in ele]

    remap_2_to_1 = {ele: ele.replace(
        "model.backbone0.", "share_bb.") for ele in backbone2}
    remap_all.update(remap_2_to_1)
    sum([remap_2_to_1[ele] in backbone1 for ele in backbone2])

    print(len(backbone1), len(backbone2), sum(
        [remap_2_to_1[ele] in backbone1 for ele in backbone2]))
    # print(backbone1)
    # print(backbone2)

    neck1 = [ele for ele in checkpoint1_keys if ("slot_bb." in ele)]
    neck2 = [ele for ele in checkpoint2_keys if ("model.group0." in ele)]
    remap_2_to_1 = {ele: ele.replace(
        "model.group0.", "slot_bb.") for ele in neck2}
    remap_all.update(remap_2_to_1)
    print(len(neck1), len(neck2), sum(
        [remap_2_to_1[ele] in neck1 for ele in neck2]))

    head1 = [ele for ele in checkpoint1_keys if ("slot_head." in ele)]
    head2 = [ele for ele in checkpoint2_keys if (
        "model.PARKING_IPM_STA." in ele)]
    remap_2_to_1 = {ele: ele.replace(
        "model.PARKING_IPM_STA.", "slot_head.") for ele in head2}
    remap_all.update(remap_2_to_1)
    print(len(head1), len(head2), sum(
        [remap_2_to_1[ele] in head1 for ele in head2]))

    others1 = [ele for ele in checkpoint1_keys if not (
        ("share_bb." in ele) or ("slot_bb." in ele) or ("slot_head." in ele))]
    others2 = [ele for ele in checkpoint2_keys if not (
        ("model.backbone0." in ele) or ("model.group0." in ele) or ("model.PARKING_IPM_STA." in ele))]

    print(len(others1), len(others2))
    print(others1)
    print(others2)

    return remap_all


def ParkingIpmSta():
    ckpt1 = "/data/ai_group/workdirs/multitask_lanenet_group/sikong/slotdetect_pointline/checkpoint/20250806-162157_ddrnet_slim23_model_last.pth"
    checkpoint1 = torch.load(ckpt1, map_location="cpu")
    checkpoint1_keys = list(checkpoint1["model"].keys())
    # print(checkpoint1_keys)

    ckpt2 = "workspace/20250818_08_19_56/checkpoint/epoch=4-step=1000_checkpoint.pth"
    checkpoint2 = torch.load(ckpt2, map_location="cpu")
    checkpoint2_keys = list(checkpoint2['state_dict'].keys())
    # print(checkpoint2_keys)

    remap_all = ParkingIpmStaRemap(checkpoint1_keys, checkpoint2_keys)
    print(len(checkpoint1_keys), len(checkpoint2_keys), len(remap_all))

    # exit(1)
    for k in checkpoint2_keys:
        if checkpoint2['state_dict'][k].shape != checkpoint1["model"][remap_all[k]].shape:
            print(k, checkpoint2['state_dict'][k].shape, checkpoint1["model"][remap_all[k]].shape,
                  checkpoint2['state_dict'][k].shape == checkpoint1["model"][remap_all[k]].shape)
        checkpoint2['state_dict'][k] = checkpoint1["model"][remap_all[k]]
    #                    lane_map_head.decoder.layers.0.sa.in_proj_weight
    # model.DRIVING_BEV_STA.head.head1.decoder.layers.0.sa.in_proj_weight
    ckpt2_edit = ckpt2.replace(".pth", "_weiwei.pth")
    print(ckpt2_edit)
    torch.save(checkpoint2, ckpt2_edit)

    bias = checkpoint2['state_dict']["model.PARKING_IPM_STA.seg_l.bn1.bias"]
    weight = checkpoint2['state_dict']["model.PARKING_IPM_STA.seg_l.bn1.weight"]

    print(bias.min(), bias.max())
    print(weight.min(), weight.max())

    bias = checkpoint1["model"]["slot_head.seg_l.bn1.bias"]
    weight = checkpoint1["model"]["slot_head.seg_l.bn1.weight"]

    print(bias.min(), bias.max())
    print(weight.min(), weight.max())


if __name__ == "__main__":
    # DrivingBevSta()
    ParkingIpmSta()
