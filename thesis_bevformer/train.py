from thesis_bevformer.utils.utils_blip import BertLMHeadModel, BertConfig, init_tokenizer
from thesis_bevformer.utils.utils_bevformer import build_bevformer, build_data_loader
from thesis_bevformer.BEVCapGen import BEVCapGen
import torch
from nuscenes.nuscenes import NuScenes

if __name__ == "__main__":
    torch.cuda.empty_cache()

    med_config_path = "./thesis_bevformer/configs/med_config.json"
    med_config = BertConfig.from_json_file(med_config_path)
    blip_lm_head = BertLMHeadModel(config=med_config)
    blip_tokenizer = init_tokenizer()

    bevformer_cfg = "./projects/configs/bevformer/bevformer_tiny.py"
    bevformer_ckpt = "./ckpts/bevformer_tiny_epoch_24.pth"
    bevformer = build_bevformer(bevformer_cfg, bevformer_ckpt)
    dataloader = build_data_loader(bevformer_cfg)
    nusc = NuScenes(version='v1.0-trainval', dataroot='./data/nuscenes', verbose=True)

    model = BEVCapGen(
        bev_encoder=bevformer,
        text_decoder=blip_lm_head,
        tokenizer=blip_tokenizer,
        bev_feature_size=256,
        decoder_hidden_size=med_config.hidden_size,
        device="cpu"
        )

    model.load_blip_decoder_ckpt("./ckpts/model_base_capfilt_large.pth")

    prompt = model.prompt

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-4, weight_decay=0.05)

    for i, data in enumerate(dataloader):
        print("NEXT__________________________________")
        
        scene_token = data["img_metas"][0].data[0][0]["scene_token"]
        caption = nusc.get("scene", scene_token)["description"]
        print("Caption:", caption)

        result = model(data, caption)
        txt = model.generate(data)
        print("Result:", result)
        print("Txt:", txt)

        """
        loss = result["loss"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
        """
        
        