#!/usr/bin/env python3

import argparse

import torch

from learn import NeuralNetwork

parser = argparse.ArgumentParser()
parser.add_argument("input_pth", type=str, help="Input .pth file")
args = parser.parse_args()

model = NeuralNetwork()
model.load_state_dict(torch.load(args.input_pth, weights_only=True))
input1 = torch.randn((1, 3), dtype=torch.float)
input2 = torch.randn((1, 80, 116), dtype=torch.float)

output_file = "calibrations/onnx/DIRCBarrel.onnx"
print(f"Will produce {output_file}")
torch.onnx.export(
    torch.jit.script(model),
    (input1, input2),
    output_file,
    input_names=["track", "dirc"],
    dynamic_axes={
        "track": {0: "N_tracks"},
    },
)
