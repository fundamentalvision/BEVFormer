import warnings
warnings.filterwarnings("ignore")

import torch
from torch import nn
import os

class BEVCapGen(nn.Module):
    def __init__(
        self,
        bev_encoder,
        text_decoder,
        tokenizer,
        bev_feature_size,
        decoder_hidden_size,
        prompt="a picture of ",
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        super().__init__()
        self.device = device

        # Freezing the BEV encoder
        self.bev_encoder = bev_encoder.to(self.device)
        for param in self.bev_encoder.parameters():
            param.requires_grad = False

        # FC layer to map BEV feature size to text decoder transformer hidden size
        self.bev_feature_mapper = nn.Sequential(
            nn.Linear(bev_feature_size, decoder_hidden_size), 
            nn.ReLU()
            ).to(self.device)

        # Text decoder and tokenizer
        self.text_decoder = text_decoder.to(self.device)
        self.tokenizer = tokenizer
        
        # Prompt
        self.prompt = prompt
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids) - 1

    def forward(self, data, caption):

        """
        data["img_metas"] = data["img_metas"].data
        data["img_metas"] = [list(data["img_metas"][0][0].values())]
        data["img"] = data["img"].data[0].to(self.device)
        data["gt_bboxes_3d"] = data["gt_bboxes_3d"].data[0]
        data["gt_labels_3d"] = [data["gt_labels_3d"].data[0][0].to(self.device)]
        """
        new_data = {}
        new_data["img_metas"] = data["img_metas"][0].data
        new_data["img"] = [data["img"][0].data[0].to(self.device)]


        bev_embeds = self.bev_encoder(return_loss=False, rescale=True, only_bev_embed=True, **new_data)
        bev_embeds = self.bev_feature_mapper(bev_embeds)
        
        bev_atts = torch.ones(bev_embeds.size()[:-1], dtype=torch.long).to(self.device)

        text = self.tokenizer(caption,
            padding="longest",
            truncation=True,
            max_length=40,
            return_tensors="pt",
        ).to(self.device)

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

        return {"logits": decoder_output.logits, "labels": decoder_targets, "loss": decoder_output.loss}

    def generate(
        self,
        data,
        max_length=30,
        min_length=10,
        top_p=0.9,
    ):
        new_data = {}
        new_data["img_metas"] = data["img_metas"][0].data
        new_data["img"] = [data["img"][0].data[0].to(self.device)]
        
        bev_embeds = self.bev_encoder(return_loss=False, rescale=True, only_bev_embed=True, **new_data)
        bev_embeds = self.bev_feature_mapper(bev_embeds)
        

        bev_atts = torch.ones(bev_embeds.size()[:-1], dtype=torch.long).to(self.device)
        model_kwargs = {
            "encoder_hidden_states": bev_embeds,
            "encoder_attention_mask": bev_atts,
        }

        prompt = [self.prompt] * 1 # batch size
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
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

    def load_blip_decoder_ckpt(self, ckpt_path):

        ckpt = torch.load(ckpt_path)
        ckpt_dict = ckpt["model"]

        self_dict = self.state_dict()
        ckpt_dict = {k: v for k, v in ckpt_dict.items() if k in self_dict}
        self_dict.update(ckpt_dict)
        self.load_state_dict(self_dict)




