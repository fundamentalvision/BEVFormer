# Copyright 2021 Toyota Research Institute.  All rights reserved.
from collections import OrderedDict

# from detectron2.config import configurable


class Task():
    def __init__(self, name, is_detection_task, is_dense_prediction_task):
        self.name = name
        self.is_detection_task = is_detection_task
        self.is_dense_prediction_task = is_dense_prediction_task


# yapf: disable
TASKS = [
    Task(
        name="box2d",
        is_detection_task=True,
        is_dense_prediction_task=False,
    ),
    Task(
        name="box3d",
        is_detection_task=True,
        is_dense_prediction_task=False,
    ),
    Task(
        name="depth",
        is_detection_task=False,
        is_dense_prediction_task=True,
    )
]
# yapf: enable

NAME_TO_TASK = OrderedDict([(task.name, task) for task in TASKS])


class TaskManager():
    #@configurable
    def __init__(self, box2d_on=False, box3d_on=False, depth_on=False):
        """
        configurable is experimental.
        """
        self._box2d_on = self._mask2d_on = self._box3d_on = self._semseg2d_on = self._depth_on = False
        tasks = []
        if box2d_on:
            tasks.append(NAME_TO_TASK['box2d'])
            self._box2d_on = True
        if box3d_on:
            tasks.append(NAME_TO_TASK['box3d'])
            self._box3d_on = True
        if depth_on:
            tasks.append(NAME_TO_TASK['depth'])
            self._depth_on = True

        if not tasks:
            raise ValueError("No task specified.")

        self._tasks = tasks

    @property
    def tasks(self):
        return self._tasks

    '''@classmethod
    def from_config(cls, cfg):
        # yapf: disable
        return OrderedDict(
            box2d_on    = cfg.MODEL.BOX2D_ON,
            box3d_on    = cfg.MODEL.BOX3D_ON,
            depth_on    = cfg.MODEL.DEPTH_ON,
        )
        # yapf: enable'''

    # Indicators that tells if each task is enabled.
    @property
    def box2d_on(self):
        return self._box2d_on

    @property
    def box3d_on(self):
        return self._box3d_on

    @property
    def depth_on(self):
        return self._depth_on

    @property
    def has_dense_prediction_task(self):
        return any([task.is_dense_prediction_task for task in self.tasks])

    @property
    def has_detection_task(self):
        return any([task.is_detection_task for task in self.tasks])

    @property
    def task_names(self):
        return [task.name for task in self.tasks]
