from pathlib import Path
from pprint import pprint
from typing import Any, Dict, Tuple, Union

import lightning as L
import torch
import torch.nn as nn
from capybara import get_curdir, now, write_metadata_into_onnx
from chameleon import calculate_flops
from otter import load_model_from_config

from . import dataset as ds
from . import model as net

DIR = get_curdir(__file__)


class Identity(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


class WarpFeatureLearning(nn.Module):

    def __init__(self, model: L.LightningModule):
        super().__init__()
        self.backbone = model.backbone
        self.head = model.head

    def forward(self, img: torch.Tensor):
        xs = self.backbone(img)
        features = self.head(xs)
        return nn.functional.normalize(features)


TORCH_TYPE_LOOKUP = {
    'float16': torch.float16,
    'float32': torch.float32,
    'float64': torch.float64,
    'int8': torch.int8,
    'int16': torch.int16,
    'int32': torch.int32,
    'int64': torch.int64,
    'uint8': torch.uint8,
    'bool': torch.bool,
    'complex32': torch.complex32,
    'complex64': torch.complex64,
    'complex128': torch.complex128,
}


def input_constructor(xs: Tuple[Dict[str, Any]]):
    return {
        k: torch.zeros(*v['shape'], dtype=TORCH_TYPE_LOOKUP[v['dtype']])
        for k, v in xs
    }


def convert_numeric_keys(input_dict):
    def try_int(key):
        # Try to convert key to an integer, return original if not possible
        try:
            return int(key)
        except ValueError:
            return key

    return {
        try_int(k): {
            try_int(inner_k): v for inner_k, v in inner_dict.items()
        } for k, inner_dict in input_dict.items()
    }


def main_classifier_torch2onnx(cfg_name: Union[str, Path]):
    model, cfg = load_model_from_config(
        cfg_name, root=DIR, stem='config', network=net)
    model = model.eval().cpu()

    warp_model_name = cfg.onnx.pop('name')
    warp_model = globals()[warp_model_name](model)
    dummy_input = input_constructor(tuple(cfg.onnx.input_shape.items()))

    if dynamic_axes := getattr(cfg.onnx, "dynamic_axes", None):
        dynamic_axes = convert_numeric_keys(dynamic_axes)

    export_name = DIR / f"{cfg_name.lower()}_{now(fmt='%Y%m%d')}_fp32"

    warp_model = warp_model.eval().cpu()
    torch.onnx.export(
        warp_model, tuple(dummy_input.values()), str(export_name) + '.onnx',
        input_names=cfg.onnx.input_names,
        output_names=cfg.onnx.output_names,
        dynamic_axes=dynamic_axes,
        **cfg.onnx.options
    )

    # To torchscript
    scripted_model = torch.jit.trace(
        warp_model, example_kwarg_inputs=dummy_input)
    torch.jit.save(scripted_model, str(export_name) + '.pt')

    flops, macs, params = calculate_flops(
        model,
        input_shape=(1, 3, *cfg.global_settings.image_size),
        print_detailed=False,
        print_results=False
    )

    additional_meta_info = getattr(cfg.onnx, 'additional_meta_info', {})
    meta_data = {
        'InputInfo': repr({k: v for k, v in cfg.onnx.input_shape.items()}),
        'FLOPs': flops,
        'MACs': macs,
        'Params': params,
        **additional_meta_info
    }

    pprint(meta_data)

    write_metadata_into_onnx(
        onnx_path=str(export_name) + '.onnx',
        out_path=str(export_name) + '.onnx',
        drop_old_meta=False,
        **meta_data
    )
