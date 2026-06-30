import torch

old_path = "last.pth.tar"
new_path = "converted_last.pth.tar"

ckpt = torch.load(old_path, map_location="cpu")

if isinstance(ckpt, dict) and "state_dict" in ckpt:
    state = ckpt["state_dict"]
else:
    state = ckpt

torch.save({"state_dict": state}, new_path)
print("saved:", new_path)