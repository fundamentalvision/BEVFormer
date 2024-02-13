from thesis_bevformer.utils.utils_blip import BertLMHeadModel, BertConfig, init_tokenizer
from thesis_bevformer.utils.utils_bevformer import build_bevformer, build_data_loader
from thesis_bevformer.BEVCapGen import BEVCapGen
import torch

if __name__ == "__main__":
    med_config_path = "./thesis_bevformer/configs/med_config.json"
    med_config = BertConfig.from_json_file(med_config_path)
    blip_lm_head = BertLMHeadModel(config=med_config)
    blip_tokenizer = init_tokenizer()

    bevformer_cfg = "./projects/configs/bevformer/bevformer_tiny.py"
    bevformer_ckpt = "./ckpts/bevformer_tiny_epoch_24.pth"
    bevformer = build_bevformer(bevformer_cfg, bevformer_ckpt)
    dataloader = build_data_loader(bevformer_cfg, mode="test")

    model = BEVCapGen(
        bev_encoder=bevformer,
        text_decoder=blip_lm_head,
        tokenizer=blip_tokenizer,
        bev_feature_size=256,
        decoder_hidden_size=med_config.hidden_size,
        device="cpu"
        )

    for i, data in enumerate(dataloader):
        with torch.no_grad():
            result = model.generate(data)
            print(result)
            break