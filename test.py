import torch

from zig_ma.Model.zigma import ZigMa

img_dim = 20
in_channels = 22

model = ZigMa(
in_channels=in_channels,
embed_dim=640,
depth=18,
img_dim=img_dim,
patch_size=1,
has_text=True,
d_context=768,
n_context_token=1,
device="cuda",
scan_type="zigzagN8",
use_pe=2,
)

x = torch.rand(10, 22, 20, 20).to("cuda")
t = torch.rand(10).to("cuda")
_context = torch.rand(10, 1, 768).to("cuda")
o = model(x, t, y=_context)
print(o.shape)
