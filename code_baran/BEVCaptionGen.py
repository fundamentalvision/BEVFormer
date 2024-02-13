import warnings

warnings.filterwarnings("ignore")

from code_baran.blip import BertLMHeadModel, BertConfig
from transformers import BertTokenizer

import torch
from torch import nn
from mmdet3d.models import build_model
from mmcv import Config
from mmdet.datasets import replace_ImageToTensor
from mmcv.runner import init_dist, load_checkpoint
import os
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader


class BEVCaptionGen(nn.Module):
    def __init__(
        self,
        med_config="./code_baran/med_config.json",
        # image_size=384,

        prompt="a picture of ",
    ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()
        self.bev_encoder = build_bevformer("./projects/configs/bevformer/bevformer_tiny.py", "./ckpts/bevformer_tiny_epoch_24.pth")

        self.tokenizer = init_tokenizer()
        med_config = BertConfig.from_json_file(med_config)
        self.text_decoder = BertLMHeadModel(config=med_config)
        self.text_decoder.cuda()

        self.prompt = prompt
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids) - 1

    def forward(self, image, caption):
        bev_embeds = self.bev_encoder(image)

        bev_atts = torch.ones(bev_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        text = self.tokenizer(caption,
            padding="longest",
            truncation=True,
            max_length=40,
            return_tensors="pt",
        ).to(image.device)

        text.input_ids[:, 0] = self.tokenizer.bos_token_id

        decoder_targets = text.input_ids.masked_fill(
            text.input_ids == self.tokenizer.pad_token_id, -100
        )
        decoder_targets[:, : self.prompt_length] = -100

        decoder_output = self.text_decoder(
            text.input_ids,
            attention_mask=text.attention_mask,
            encoder_hidden_states=bev_embeds,
            encoder_attention_mask=bev_atts,
            labels=decoder_targets,
            return_dict=True,
        )
        loss_lm = decoder_output.loss

        return loss_lm

    def generate(
        self,
        data,
        max_length=30,
        min_length=10,
        top_p=0.9,
    ):
        data["img_metas"] = data["img_metas"][0].data
        data["img"] = [data["img"][0].data[0].cuda()]
        
        bev_embeds = self.bev_encoder(return_loss=False, rescale=True, **data)
        mlp = nn.Sequential(nn.Linear(256, 768), nn.ReLU()).cuda()
        bev_embeds = mlp(bev_embeds)
        bev_embeds = bev_embeds.permute(1, 0, 2)

        bev_atts = torch.ones(bev_embeds.size()[:-1], dtype=torch.long).cuda()
        model_kwargs = {
            "encoder_hidden_states": bev_embeds,
            "encoder_attention_mask": bev_atts,
        }

        prompt = [self.prompt] * 1 # batch size
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        input_ids[:, 0] = self.tokenizer.bos_token_id
        input_ids = input_ids[:, :-1]


        # nucleus sampling
        outputs = self.text_decoder.generate(
            input_ids=input_ids,
            max_length=max_length,
            min_length=min_length,
            do_sample=True,
            top_p=top_p,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            repetition_penalty=1.1,
            **model_kwargs
        )

        captions = []
        for output in outputs:
            caption = self.tokenizer.decode(output, skip_special_tokens=True)
            captions.append(caption[len(self.prompt) :])
        return captions


def init_tokenizer():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.add_special_tokens({"bos_token": "[DEC]"})
    tokenizer.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
    return tokenizer

def build_bevformer(config, checkpoint):

    cfg = Config.fromfile(config)

    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # set tf32
    if cfg.get('close_tf32', False):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    init_dist("pytorch", **cfg.dist_params)

    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')

    model.CLASSES = checkpoint['meta']['CLASSES']
    # palette for visualization in segmentation tasks
    model.PALETTE = checkpoint['meta']['PALETTE']
    model = model.cuda()

    return model
