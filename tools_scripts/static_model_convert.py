
import torch
import os
def StaticModelCenvert():
    checkoutpoint_path = "/data/ai_group/workdirs/gpal_neural_network_group/airflow_workspace/gpal_neural_network_one_node_traning_job_on_airflow_20251011_11_59_23/checkpoint/epoch=0-step=30000_checkpoint.pth"
    checkpoint = torch.load(checkoutpoint_path, map_location="cpu")
    save_path = '/data/ai_group/workdirs/multitask_lanenet_group/wujianlong/multitask_lanenet/work_dir/gpal_neural_network_1wclips_1013/model/ep001.pth'  
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    old_framework_pth = {}
    
    checkpoint_keys = list(checkpoint['state_dict'].keys())
    
    for k in checkpoint_keys:
        if "model.backbone0." in k:
            new_k = k.replace("model.backbone0.", "backbone.")
            old_framework_pth[new_k] = checkpoint['state_dict'][k]
        elif "model.neck0." in k:
            new_k = k.replace("model.neck0.", "neck.")
            old_framework_pth[new_k] = checkpoint['state_dict'][k]
        elif "model.transformer." in k:
            new_k = k.replace("model.transformer.", "vt.")
            old_framework_pth[new_k] = checkpoint['state_dict'][k]
        elif "model.DRIVING_BEV_STA.head.head1." in k:
            new_k = k.replace("model.DRIVING_BEV_STA.head.head1.", "lane_map_head.")
            old_framework_pth[new_k] = checkpoint['state_dict'][k]
        else:
            raise ValueError(f"Unknown key {k}")
    torch.save(old_framework_pth, save_path)

if __name__ == "__main__":
    StaticModelCenvert()