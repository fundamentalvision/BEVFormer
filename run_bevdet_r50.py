# Evaluate Bevformer_FP16
from pathlib import Path
import sys
import time
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model
from nuscenes.nuscenes import NuScenes
from eval_benchmark import evaluate, save_timing

ROOT = Path(__file__).resolve().parent


def main():
    # We can copy most code from bevformer mini
    cfg = Config.fromfile(str(Path(sys.prefix) / 'src/BEVDet/configs/bevdet/bevdet-r50.py'))
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.data.test.update(
        data_root=str(ROOT / 'data/nuscenes'),
        ann_file=str(ROOT / 'data/nuscenes/bevdet_infos_mini_val.pkl'),
        test_mode=True,
    )
    dataset = build_dataset(cfg.data.test)
    loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False)

    # Load the pretrained weights and run on the GPU.
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = torch.load(ROOT / 'ckpts/bevdet-r50.pth', map_location='cpu', weights_only=False)
    weights = {key.removeprefix('module.'): value for key, value in checkpoint['state_dict'].items()}
    model.load_state_dict(weights, strict=True)
    del checkpoint, weights
    model.CLASSES = dataset.CLASSES
    torch.cuda.set_device(0)
    model = MMDataParallel(model.cuda(), device_ids=[0]).eval()

    # Keep frames in order so BEVFormer can carry its BEV history across time.
    predictions = []
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for data in loader:
            predictions.extend(model(return_loss=False, rescale=True, **data))

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    # Export global-coordinate detections and use the shared six-class evaluator.
    result_files, _ = dataset.format_results(predictions, jsonfile_prefix=str(ROOT / 'outputs/bevdet-r50'))
    save_timing(elapsed, len(dataset), "GPU", str(Path(result_files["pts_bbox"]).parent / "timing.json"))
    nusc = NuScenes(version='v1.0-mini', dataroot=str(ROOT / 'data/nuscenes'), verbose=False,)
    evaluate(nusc, 'bevdet-r50', result_files['pts_bbox'])


if __name__ == '__main__':
    main()
