import os
import sys
import cv2
import numpy as np
import pickle
import argparse
from typing import List, Optional, Tuple, Dict
from tqdm import tqdm
import torch
import yaml
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# if repo_root not in sys.path:
sys.path.insert(0, repo_root)
from gpal_lightning.neural_network.global_config import GlobalConfig
from gpal_nn.models.transformers.bevformer.view_transformer import SingleBevFormerViewTransformer
from tools_scripts.driving_bev_sta.create_images_grid import PreproModule


def load_pkl(pkl_path):
	with open(pkl_path, "rb") as f:
		data = pickle.load(f)
	return list(data)


def ensure_dir(path: str):
	os.makedirs(path, exist_ok=True)


def resolve_img_path(img_path: str, image_root) -> str:
	if os.path.isabs(img_path):
		return img_path
	if image_root:
		return os.path.join(image_root, img_path)
	return img_path


def read_image(img_fp: str):
	# STA模型输入是RGB,校准集应对齐，Read image in BGR format and convert to RGB float32
	img = cv2.cvtColor(cv2.imread(img_fp, cv2.IMREAD_COLOR).astype(np.float32), cv2.COLOR_BGR2RGB)
	if img is None:
		raise FileNotFoundError(f"Failed to read image: {img_fp}")
	return img


def save_npy(arr: np.ndarray, out_fp: str):
	ensure_dir(os.path.dirname(out_fp))
	np.save(out_fp, arr)


def entry_get(entry: dict, key_path: List[str], default=None):
	cur = entry
	for k in key_path:
		if not isinstance(cur, dict) or (k not in cur):
			return default
		cur = cur[k]
	return cur


def extract_img_paths(entry: dict,
					  key_30: str = "img_front_30",
					  key_120: str = "img_front_120"):
	sensor = entry.get("sensor", {}) if isinstance(entry, dict) else {}
	item_30 = sensor.get(key_30, {})
	item_120 = sensor.get(key_120, {})
	path_30 = item_30.get("img_path")
	path_120 = item_120.get("img_path")
	if path_30 is None and "image_30" in sensor:
		path_30 = sensor.get("image_30", {}).get("img_path")
	if path_120 is None and "image_120" in sensor:
		path_120 = sensor.get("image_120", {}).get("img_path")
	return path_30, path_120


def extract_intrinsics_and_dists(entry: dict, cam_keys: List[str]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
	"""Pull 3x3 K and (5,) dist per camera from the pkl entry (see abel/ann.txt example)."""
	sensor = entry.get("sensor", {}) if isinstance(entry, dict) else {}
	Ks, dists = [], []
	for k in cam_keys:
		cam = sensor.get(k, {})
		intr = cam.get("intr", {})
		K = intr.get("K")
		dist = intr.get("dist")
		if K is None or dist is None:
			raise KeyError(f"Missing intrinsics/distortion for camera '{k}' in entry")
		Ks.append(np.array(K, dtype=np.float32))
		dists.append(np.array(dist, dtype=np.float32).reshape(-1))
	return Ks, dists


def build_ego2imgs(entry: dict,
				   cam_keys: List[str],
				   ori_shape: Tuple[int, int],
				   dst_h: int,
				   dst_w: int,
				   cut_start_h: int) -> np.ndarray:
	"""Compute ego2imgs (1, N_cam, 4, 4) consistent with dataset:
	- letterbox: scale by s=min(dst_w/ori_w, dst_h/ori_h); pad: left=(dst_w-w')/2, top=(dst_h-h')/2
	- crop: K[1,2] -= cut_start_h
	- ego2img = K' @ [R|T]
	"""
	ori_h, ori_w = ori_shape
	scale = min(dst_w / float(ori_w), dst_h / float(ori_h))
	new_w = int(round(ori_w * scale))
	new_h = int(round(ori_h * scale))
	pad_w = dst_w - new_w
	pad_h = dst_h - new_h
	pad_left = pad_w // 2
	pad_top = pad_h // 2

	sensor = entry.get("sensor", {}) if isinstance(entry, dict) else {}
	mats = []
	for k in cam_keys:
		cam = sensor.get(k, {})
		K = np.array(cam.get("intr", {}).get("K"), dtype=np.float32).copy()  # raw 3x3
		extr = cam.get("extr", {})
		R = np.array(extr.get("rot"), dtype=np.float32)  # 3x3
		T = np.array(extr.get("T"), dtype=np.float32).reshape(3, 1)  # 3x1

		K_resized = K.copy()
		K_resized[0, :] *= scale
		K_resized[1, :] *= scale
		K_resized[0, 2] += pad_left
		K_resized[1, 2] += pad_top
		K_resized[1, 2] -= float(cut_start_h)

		ext34 = np.concatenate([R, T], axis=1)  # 3x4
		ego2img_3x4 = K_resized @ ext34
		ego2img_4x4 = np.concatenate([ego2img_3x4, np.array([[0, 0, 0, 1]], dtype=np.float32)], axis=0)
		mats.append(ego2img_4x4)
	ego2imgs = np.stack(mats, axis=0)  # (N_cam, 4, 4)
	ego2imgs = ego2imgs[None, ...]  # (1, N_cam, 4, 4)
	return ego2imgs


def gen_images_grid(batch_size: int,
					Ks: List[np.ndarray],
					dists: List[np.ndarray],
					ori_shape: Tuple[int, int],
					dst_h: int,
					dst_w: int,
					cut_start_h: int) -> np.ndarray:
	"""Use tools_scripts/driving_bev_sta/create_images_grid.PreproModule.export_input to build images_grid."""

	if len(Ks) != 2 or len(dists) != 2:
		raise ValueError("images_grid requires exactly 2 cameras (K/dist)")

	K1, K2 = Ks[0].copy(), Ks[1].copy()
	d1, d2 = dists[0].copy(), dists[1].copy()
	pm = PreproModule()
	grid = pm.export_input(batch_size, K1, d1, K2, d2, ori_shape, dst_h, dst_w, cut_start_h)
	return grid.detach().cpu().numpy()


def ensure_calib_dirs(save_dir: str):
	for sd in [
		"bev_pillar_counts",
		"images_grid",
		"queries_rebatch_grid",
		"reference_points_rebatch",
		"restore_bev_grid",
	]:
		ensure_dir(os.path.join(save_dir, sd))


def build_view_transformer(config_yaml: str):
	"""Instantiate SingleBevFormerViewTransformer from YAML. Returns (vt_module, device)."""

	with open(config_yaml, "r") as f:
		cfg = yaml.safe_load(f)
	global_cfg = GlobalConfig(cfg)
	transformer_cfg = cfg.get("Transformer", {}).get("transformer_config", {})
	vt = SingleBevFormerViewTransformer(global_cfg, transformer_cfg)
	device = torch.device("cpu")
	vt.to(device)
	vt.eval()
	return vt, device


def default_out_name(idx: int, img_path: Optional[str]) -> str:
	if img_path:
		base = os.path.basename(img_path)
		stem, _ = os.path.splitext(base)
		if stem:
			return stem
	return f"{idx:06d}"


def convert(pkl_file: str,
			save_dir: str,
			image_root: Optional[str] = None,
			key_30: str = "img_front_30",
			key_120: str = "img_front_120",
			subdir_30: str = "img_30",
			subdir_120: str = "img_120",
			config_yaml: Optional[str] = None,
			dst_h: int = 540,
			dst_w: int = 960,
			cut_start_h: int = 28,
			cam_order: Optional[List[str]] = None,
			bev_real2aug: Optional[np.ndarray] = None,
			limit: Optional[int] = 200):
	data = load_pkl(pkl_file)
	out_30_dir = os.path.join(save_dir, subdir_30)
	out_120_dir = os.path.join(save_dir, subdir_120)
	ensure_calib_dirs(save_dir)
	ensure_dir(out_30_dir)
	ensure_dir(out_120_dir)

	vt = None
	vt_device = None
	if config_yaml is None:
		config_yaml = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../configs_for_develop/driving_bev_sta_config.yaml"))
	vt, vt_device = build_view_transformer(config_yaml)
	# Cache ref3d for bs=1
	ref2d, ref3d = vt.export_reference_points(bs=1, device=vt_device)
	# Default bev_real2aug to identity if not provided
	if bev_real2aug is None:
		bev_real2aug = np.eye(4, dtype=np.float32)
	if cam_order is None:
		cam_order = [key_30, key_120]

	errors = []
	for idx, entry in enumerate(tqdm(data[0:limit], desc="Converting frames"), start=0):
		try:
			rel_30, rel_120 = extract_img_paths(entry, key_30, key_120)
			if rel_30 is None and rel_120 is None:
				raise KeyError(f"Missing both {key_30} and {key_120} paths at idx={idx}")

			# 30
			if rel_30 is not None:
				fp30 = resolve_img_path(rel_30, image_root)
				img30 = read_image(fp30)
				img30 = img30[np.newaxis, ...]
				out30 = os.path.join(out_30_dir, f"{idx:04d}.npy")
				save_npy(img30, out30)

			# 120
			if rel_120 is not None:
				fp120 = resolve_img_path(rel_120, image_root)
				img120 = read_image(fp120)
				img120 = img120[np.newaxis, ...]
				out120 = os.path.join(out_120_dir, f"{idx:04d}.npy")
				save_npy(img120, out120)

				Ks, dists = extract_intrinsics_and_dists(entry, cam_order)
			# images_grid
			if rel_30 is not None:
				# both cams should share same ori shape
				ori_h, ori_w = int(img30.shape[1]), int(img30.shape[2])
			elif rel_120 is not None:
				ori_h, ori_w = int(img120.shape[1]), int(img120.shape[2])
			else:
				raise RuntimeError("No image present to infer original shape")
			images_grid = gen_images_grid(
				batch_size=1,
				Ks=Ks,
				dists=dists,
				ori_shape=(ori_h, ori_w),
				dst_h=dst_h,
				dst_w=dst_w,
				cut_start_h=cut_start_h,
			)  # shape [2, dst_h-cut, dst_w, 2]
			save_npy(images_grid, os.path.join(save_dir, "images_grid", f"{idx:04d}.npy"))

			# ego2imgs aligned with letterboxed + cropped image plane
			ego2imgs = build_ego2imgs(entry, cam_order, (ori_h, ori_w), dst_h, dst_w, cut_start_h)  # (1, N_cam, 4, 4)

			# im_shape should match rectified and cropped input size
			im_shape = (dst_h - cut_start_h, dst_w)

			# point_sampling -> queries_rebatch_grid, restore_bev_grid, reference_points_rebatch, bev_pillar_counts
			bev_real2aug_t = torch.from_numpy(bev_real2aug).to(vt_device)
			(
				reference_points_rebatch,
				queries_rebatch_grid,
				restore_bev_grid,
				bev_pillar_counts,
			) = vt.point_sampling(
				reference_points=ref3d,
				pc_range=vt.pc_range,
				img_metas={"ego2imgs": torch.from_numpy(ego2imgs).to(vt_device)},
				im_shape=im_shape,
				bev_real2aug=bev_real2aug_t,
			)
			save_npy(queries_rebatch_grid.detach().cpu().numpy(), os.path.join(save_dir, "queries_rebatch_grid", f"{idx:04d}.npy"))
			save_npy(reference_points_rebatch.detach().cpu().numpy(), os.path.join(save_dir, "reference_points_rebatch", f"{idx:04d}.npy"))
			save_npy(restore_bev_grid.detach().cpu().numpy(), os.path.join(save_dir, "restore_bev_grid", f"{idx:04d}.npy"))
			save_npy(bev_pillar_counts.detach().cpu().numpy(), os.path.join(save_dir, "bev_pillar_counts", f"{idx:04d}.npy"))

		except Exception as e:
			errors.append((idx, str(e)))

	if errors:
		print(f"Completed with {len(errors)} errors:")
		for i, msg in errors[:10]:
			print(f"  idx={i}: {msg}")
		if len(errors) > 10:
			print(f"  ... and {len(errors)-10} more")
	else:
		print("Completed without errors.")


def parse_args(argv=None):
	parser = argparse.ArgumentParser(description="Convert PKL frames to NPY images and generate calibration arrays")
	parser.add_argument("--pkl-file", default="/data/ai_group/datasets/multiview_lane_det/lane_pkl/calib_data_20250915.pkl", help="Path to input PKL file (list of frames)")
	# parser.add_argument("--pkl-file", default="/data/ai_group/datasets/multiview_lane_det/lane_pkl/1f_from_20250906_split_merge_val.pkl", help="Path to input PKL file (list of frames)")
	parser.add_argument("--save-dir", default="", help="Output directory containing img_30/img_120 and calibration subfolders")
	parser.add_argument("--image-root", default="/data/dp_group/process-prod-bucket/business_datasets/lane_bev_data", help="Root directory to prepend to relative image paths in PKL")
	parser.add_argument("--key-30", default="img_front_30")
	parser.add_argument("--key-120", default="img_front_120")
	parser.add_argument("--subdir-30", default="img_30")
	parser.add_argument("--subdir-120", default="img_120")
	# parser.add_argument("--config-yaml", default="/data/ai_group/workdirs/gpal_neural_network_group/airflow_workspace/gpal_neural_network_one_node_traning_job_on_airflow_20251008_06_18_22/onnx-config.yaml", help="Path to transformer config YAML for view transformer")
	parser.add_argument("--config-yaml", default="/data/ai_group/workdirs/gpal_neural_network_group/airflow_workspace/gpal_neural_network_one_node_traning_job_on_airflow_20251011_11_59_23/config.yaml")
	parser.add_argument("--dst-h", type=int, default=540)
	parser.add_argument("--dst-w", type=int, default=960)
	parser.add_argument("--cut-start-h", type=int, default=28, help="Top crop rows used in images_grid generation")
	parser.add_argument("--cam-order", nargs="*", default=None, help="e.g. img_front_30 img_front_120")
	parser.add_argument("--limit", default=None, help="Max number of frames to process")
	return parser.parse_args(argv)


if __name__ == "__main__":
	args = parse_args()
	convert(
		pkl_file=args.pkl_file,
		save_dir=args.save_dir,
		image_root=args.image_root,
		key_30=args.key_30,
		key_120=args.key_120,
		subdir_30=args.subdir_30,
		subdir_120=args.subdir_120,
		config_yaml=args.config_yaml,
		dst_h=args.dst_h,
		dst_w=args.dst_w,
		cut_start_h=args.cut_start_h,
		cam_order=args.cam_order,
		limit=args.limit,
	)