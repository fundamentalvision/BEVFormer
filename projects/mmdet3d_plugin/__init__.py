# Register the shared framework's bbox components before BEVFormer's overrides.
import mmdet3d.core

from .core.bbox.assigners.hungarian_assigner_3d import HungarianAssigner3D
from .core.bbox.coders.nms_free_coder import NMSFreeCoder
from .core.bbox.match_costs import BBox3DL1Cost
from .core.evaluation.eval_hooks import CustomDistEvalHook
from .datasets.pipelines import (
  PhotoMetricDistortionMultiViewImage, PadMultiViewImage, 
  NormalizeMultiviewImage,  CustomCollect3D)
from .models.utils import *
from .models.opt.adamw import AdamW2
from .bevformer import *
try:
    from .dd3d import *
except ImportError as _e:
    # Guarded by patches/bevformer/12_plugin_init_guard.sh
    # Reason: dd3d/ subpackage may pull Detectron2 (un-installed in v5).
    # Normally patch 09 empties dd3d/__init__.py (no-op), this guards against
    # patch-application atomicity / future changes adding dd3d imports.
    import warnings
    warnings.warn(f'dd3d subpackage unavailable: {_e}')
