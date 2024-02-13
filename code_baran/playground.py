"""
from transformers import AutoTokenizer, DistilBertModel
import torch

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")

inputs = tokenizer(
    ["seven", "7", "today"],
    return_tensors="pt", 
)
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
print(last_hidden_states)
"""

"""
import os
import torch.distributed.launch

launch_file_path = os.path.abspath(torch.distributed.launch.__file__)
print(launch_file_path)
"""

from nuscenes.nuscenes import NuScenes
nusc = NuScenes(version='v1.0-trainval', dataroot='./data/nuscenes', verbose=True)
# nusc.list_scenes()
dsc = nusc.get("scene", "c3ab8ee2c1a54068a72d7eb4cf22e43d")["description"]
print(dsc)
