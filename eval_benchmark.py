# Shared evaluation for every BEV variant: results JSON (nuScenes format) in -> mAP and mATE out.
# Run this file directly to check it on the ground truth (should print mAP 1.0, mATE 0.0).
#
# Metrics (averaged over the 6 CLASSES below):
#   mAP  = mean Average Precision: did we find the objects without false alarms? 0-1, higher is better.
#          a prediction counts as correct if its center is within 0.5 / 1 / 2 / 4 m of a real object
#          (on the ground, x-y only); AP is averaged over those 4 thresholds.
#   mATE = mean Average Translation Error: for correct matches (2 m threshold), how far off is the
#          center, in metres? lower is better. if nothing matches, nuScenes reports 1.0 as a placeholder.
#   runtime  = time to run the method's whole script, if it saved a timing.json; note the device,
#          CPU and GPU times are not directly comparable.

import json
import os
from contextlib import redirect_stdout, redirect_stderr

import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.evaluate import DetectionEval
from nuscenes.eval.detection.utils import category_to_detection_name

# the 6 classes our 2D detector (COCO) can produce; we only score these
CLASSES = ["car", "truck", "bus", "pedestrian", "bicycle", "motorcycle"]


def val_sample_tokens(nusc):
    # the 81 samples in the mini_val split
    val_scenes = create_splits_scenes()["mini_val"]
    tokens = []
    for sample in nusc.sample:
        scene = nusc.get("scene", sample["scene_token"])
        if scene["name"] in val_scenes:
            tokens.append(sample["token"])
    return tokens


# get the ego pose for a given sample
def ego_pose(nusc, sample_token):
    # where the car is in the world at this sample: position t and rotation q
    sample = nusc.get("sample", sample_token)
    lidar = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    pose = nusc.get("ego_pose", lidar["ego_pose_token"])
    return np.array(pose["translation"]), Quaternion(pose["rotation"])


# convert a single box from the ego frame to the global frame
def ego_box_to_global(box, t, q, sample_token):
    # one box: ego frame -> global frame, in the format the nuScenes evaluator wants
    center = q.rotate(np.array(box["center"])) + t
    rotation = q * Quaternion(axis=[0, 0, 1], angle=box["yaw"])
    return {
        "sample_token": sample_token,
        "translation": center.tolist(),
        "size": list(box["size"]),
        "rotation": rotation.elements.tolist(),
        "velocity": [0.0, 0.0],      # we don't estimate velocity
        "detection_name": box["name"],
        "detection_score": box["score"],
        "attribute_name": "",        # we don't estimate attributes
    }


def save_results(results, path):
    # results = {sample_token: [result box, ...]} in the nuScenes format (global frame),
    # with an entry for every mini_val sample (empty list if nothing was detected)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    meta = {"use_camera": False, "use_lidar": False, "use_radar": False,   # required by the format,
            "use_map": False, "use_external": False}                       # does not affect scores
    with open(path, "w") as f:
        json.dump({"meta": meta, "results": results}, f)


def save_timing(total_sec, n_samples, device, path):
    # runtime of a method, saved next to its results.json so evaluate() can report it
    # total_sec: time to run the whole script (loading data and model, all samples)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"total_sec": total_sec, "sec_per_sample": total_sec / n_samples, "device": device}, f)


# evaluate a given method
def evaluate(nusc, variant_name, results_path):
    # results_path: a results JSON already in the nuScenes format (global frame), e.g. from save_results.
    # each box: {"sample_token", "translation", "size", "rotation", "velocity",
    #            "detection_name", "detection_score", "attribute_name"}

    # run the official nuScenes evaluator
    # (the devkit always prints its full summary, so its output is silenced; we print only mAP and mATE)
    out_dir = os.path.join("outputs", variant_name)
    with open(os.devnull, "w") as quiet, redirect_stdout(quiet), redirect_stderr(quiet):
        evaluator = DetectionEval(nusc, config=config_factory("detection_cvpr_2019"), result_path=results_path,
                                  eval_set="mini_val", output_dir=out_dir, verbose=False)
        metrics = evaluator.main(plot_examples=0, render_curves=False)

    # average mAP and mATE over our 6 classes
    aps, ates = [], []
    for c in CLASSES:
        aps.append(np.mean(list(metrics["label_aps"][c].values())))   # AP averaged over 0.5/1/2/4 m
        ates.append(metrics["label_tp_errors"][c]["trans_err"])       # position error in metres
    mAP = float(np.mean(aps))
    mATE = float(np.nanmean(ates))

    # runtime, if the method saved a timing.json next to its results (see save_timing)
    timing_path = os.path.join(os.path.dirname(results_path), "timing.json")
    sec_per_sample, speed = None, "no timing"
    if os.path.exists(timing_path):
        with open(timing_path) as f:
            timing = json.load(f)
        sec_per_sample = timing["sec_per_sample"]
        speed = f"{timing['total_sec']:.0f} s total, {sec_per_sample:.2f} s/sample ({timing['device']})"

    print(f"[{variant_name}] mAP {mAP:.4f} | mATE {mATE:.3f} m | {speed}")
    return {"mAP": mAP, "mATE": mATE, "sec_per_sample": sec_per_sample}


if __name__ == "__main__":
    nusc = NuScenes(version="v1.0-mini", dataroot="v1.0-mini", verbose=False)

    # ========== GROUND TRUTH (sanity check: should give mAP 1.0, mATE 0.0) ==========
    # the real annotations (already global) written as result boxes
    results = {}
    for token in val_sample_tokens(nusc):
        results[token] = []
        for ann_token in nusc.get("sample", token)["anns"]:
            ann = nusc.get("sample_annotation", ann_token)
            name = category_to_detection_name(ann["category_name"])
            if name is None:
                continue
            if ann["num_lidar_pts"] + ann["num_radar_pts"] == 0:   # the evaluator ignores these, so skip them
                continue
            results[token].append({
                "sample_token": token,
                "translation": ann["translation"],
                "size": ann["size"],
                "rotation": ann["rotation"],
                "velocity": [0.0, 0.0],
                "detection_name": name,
                "detection_score": 1.0,
                "attribute_name": "",
            })
    path = "outputs/ground_truth/results.json"
    save_results(results, path)
    evaluate(nusc, "ground_truth", path)

    # ========== CAM-B: DETECTOR ON THE BEV IMAGE (null baseline) ==========
    # results come from running deterministic_bev_cam.py
    evaluate(nusc, "cam_bev_detect", "outputs/cam_bev_detect/results.json")

    # ========== CAM-G: DETECTOR ON EACH CAMERA, PROJECTED TO THE GROUND ==========
    # results come from running cam_ground_project.py
    evaluate(nusc, "cam_ground_project", "outputs/cam_ground_project/results.json")
