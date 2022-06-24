from data import VIL, multibatch_collate_fn
from transform import TrainTransform, TestTransform

from options import OPTION as opt

import torch.utils.data as data

input_dim = opt.input_size

train_transformer = TrainTransform(size=input_dim)
test_transformer = TestTransform(size=input_dim)

trainset = VIL( 
    train=True,
    sampled_frames=opt.sampled_frames, # 对每个视频采样9帧,最多10帧，采样全部
    transform=train_transformer,
    max_skip=opt.max_skip[0], # 帧之间最大跳步,这里取索引0是因为设置列表可将不同采样帧数数据集放一块训练
    samples_per_video=opt.samples_per_video # 对每个视频采样2次，也就是相当于训练2轮
)
testset = VIL( # opt.valset
    train=False,
    transform=test_transformer,
    samples_per_video=1
)
trainloader = data.DataLoader(trainset, batch_size=opt.train_batch, shuffle=True, num_workers=opt.workers,collate_fn=multibatch_collate_fn, drop_last=True)