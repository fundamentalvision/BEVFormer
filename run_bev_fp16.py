# Evaluate Bevformer_FP16
from pathlib import Path

import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import wrap_fp16_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from nuscenes.nuscenes import NuScenes

import projects.mmdet3d_plugin  # Register BEVFormer and its data pipeline.
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from eval_benchmark import evaluate

ROOT = Path(__file__).resolve().parent


def main():
    # We can copy most code from bevformer mini
    cfg = Config.fromfile(str(ROOT / 'projects/configs/bevformer_fp16/bevformer_tiny_fp16.py'))
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.data.test.update(
        data_root=str(ROOT / 'data/nuscenes'),
        ann_file=str(ROOT / 'data/nuscenes/nuscenes_infos_temporal_val.pkl'),
        test_mode=True,
    )
    dataset = build_dataset(cfg.data.test)
    loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False)

    # Load the pretrained weights and run on the GPU.
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = torch.load(ROOT / 'ckpts/bevformer_tiny_epoch_24.pth', map_location='cpu', weights_only=False)
    weights = {key.removeprefix('module.'): value for key, value in checkpoint['state_dict'].items()}
    model.load_state_dict(weights, strict=True)
    del checkpoint, weights
    model.CLASSES = dataset.CLASSES
    torch.cuda.set_device(0)
    wrap_fp16_model(model)
    model = MMDataParallel(model.cuda(), device_ids=[0]).eval()

    # Keep frames in order so BEVFormer can carry its BEV history across time.
    predictions = []
    with torch.no_grad():
        for data in loader:
            predictions.extend(model(return_loss=False, rescale=True, **data))

    # Export global-coordinate detections and use the shared six-class evaluator.
    result_files, _ = dataset.format_results(predictions, jsonfile_prefix=str(ROOT / 'outputs/bevformer_tiny_fp16'))
    nusc = NuScenes(version='v1.0-mini', dataroot=str(ROOT / 'data/nuscenes'), verbose=False,)
    evaluate(nusc, 'bevformer_tiny_fp16', result_files['pts_bbox'])


if __name__ == '__main__':
    main()
