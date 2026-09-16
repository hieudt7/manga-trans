"""Export vietocr's vgg_transformer recogniser to ONNX for koharu-ml/src/vietocr.

    python scripts/export_vietocr_to_onnx.py ~/.cache/koharu/vietocr

Needs: vietocr torch torchvision onnx onnxruntime onnxscript (torch >= 2.5).

Writes vietocr_encoder.onnx, vietocr_decoder.onnx and vietocr.json. Each step
below that looks unusual is there because the plain export produced a graph
that loaded but read garbage or refused other shapes.
"""

import json
import math
import os
import sys
import warnings

import onnx
import torch
from torch import nn

import vietocr.model.backbone.vgg as vgg
from vietocr.tool.config import Cfg
from vietocr.tool.translate import build_model
from vietocr.tool.utils import download_weights

warnings.filterwarnings("ignore")
out = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "models/vietocr")
os.makedirs(out, exist_ok=True)


# ONNX rejects negative indices in a permutation, and vietocr's VGG ends with
# permute(-1, 0, 1). On a 3-d tensor that is permute(2, 0, 1).
def _vgg_forward(self, x):
    conv = self.last_conv_1x1(self.dropout(self.features(x)))
    return conv.transpose(2, 3).flatten(2).permute(2, 0, 1)


vgg.Vgg.forward = _vgg_forward

cfg = Cfg.load_config_from_name("vgg_transformer")
cfg["device"] = "cpu"
cfg["cnn"]["pretrained"] = False
model, _ = build_model(cfg)
# build_model only builds the architecture. Without this the export is a
# randomly initialised model that reads every line as `Q%%%%ẳẳẳ`.
weights = download_weights(cfg["weights"]) if cfg["weights"].startswith("http") else cfg["weights"]
model.load_state_dict(torch.load(weights, map_location="cpu"))
model.eval()


class Encoder(nn.Module):
    """A one-line strip -> the sequence the decoder attends to."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, img):
        return self.m.transformer.forward_encoder(self.m.cnn(img))


class Decoder(nn.Module):
    """The prefix so far + the encoder memory -> scores for the next character.

    The causal mask comes in as an input. vietocr builds it from
    tgt.shape[0] as a Python int, which bakes the sequence length into the
    graph. tgt_is_causal=True stops PyTorch checking the mask with torch.all(),
    a data-dependent branch that blocks symbolic shapes entirely.
    """

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, tgt, memory, tgt_mask):
        t = self.m.transformer
        x = t.pos_enc(t.embed_tgt(tgt) * math.sqrt(t.d_model))
        y = t.transformer.decoder(
            x, memory, tgt_mask=tgt_mask, tgt_is_causal=True, memory_is_causal=False
        )
        return t.fc(y.transpose(0, 1))


img = torch.rand(1, 3, 32, 160)
with torch.no_grad():
    memory = Encoder(model)(img)

width = torch.export.Dim("width", min=4, max=512)
steps = torch.export.Dim("steps", min=1, max=256)
length = torch.export.Dim("length", min=1, max=128)

# The dynamo exporter: the legacy tracer fixed every attention reshape to the
# length seen while tracing, so any other width failed at runtime.
torch.onnx.export(
    Encoder(model),
    (img,),
    f"{out}/vietocr_encoder.onnx",
    input_names=["image"],
    output_names=["memory"],
    dynamic_shapes={"img": {3: width}},
    opset_version=18,
    dynamo=True,
)

tgt = torch.ones(3, 1, dtype=torch.long)
mask = torch.triu(torch.full((3, 3), float("-inf")), diagonal=1)
torch.onnx.export(
    Decoder(model),
    (tgt, memory, mask),
    f"{out}/vietocr_decoder.onnx",
    input_names=["tokens", "memory", "mask"],
    output_names=["logits"],
    dynamic_shapes={
        "tgt": {0: length},
        "memory": {0: steps},
        "tgt_mask": {0: length, 1: length},
    },
    opset_version=18,
    dynamo=True,
)

with open(f"{out}/vietocr.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "vocab": cfg["vocab"],
            "pad": 0,
            "sos": 1,
            "eos": 2,
            "mask": 3,
            "first_char_id": 4,
            "image_height": cfg["dataset"]["image_height"],
            "image_min_width": cfg["dataset"]["image_min_width"],
            "image_max_width": cfg["dataset"]["image_max_width"],
            "width_round_to": 10,
        },
        f,
        ensure_ascii=False,
    )

# The dynamo exporter writes weights to a side file; fold them back in so each
# model is one path to download and load.
for name in ("vietocr_encoder", "vietocr_decoder"):
    path = f"{out}/{name}.onnx"
    onnx.save(onnx.load(path), path, save_as_external_data=False)
    if os.path.exists(path + ".data"):
        os.remove(path + ".data")
    print(f"{path}  {os.path.getsize(path) // 1024 // 1024} MB")
