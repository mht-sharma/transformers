# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
import warnings
from typing import Dict, Optional, Union

from ..models.auto.configuration_auto import AutoConfig
from ..utils.quantization_config import (
    AqlmConfig,
    AwqConfig,
    BitsAndBytesConfig,
    EetqConfig,
    FbgemmFp8Config,
    GPTQConfig,
    HqqConfig,
    QuantizationConfigMixin,
    QuantizationMethod,
    QuantoConfig,
    TorchAoConfig,
)
from .quantizer_aqlm import AqlmHfQuantizer
from .quantizer_awq import AwqQuantizer
from .quantizer_bnb_4bit import Bnb4BitHfQuantizer
from .quantizer_bnb_8bit import Bnb8BitHfQuantizer
from .quantizer_eetq import EetqHfQuantizer
from .quantizer_fbgemm_fp8 import FbgemmFp8HfQuantizer
from .quantizer_gptq import GptqHfQuantizer
from .quantizer_hqq import HqqHfQuantizer
from .quantizer_quanto import QuantoHfQuantizer
from .quantizer_torchao import TorchAoHfQuantizer


AUTO_QUANTIZER_MAPPING = {
    "awq": AwqQuantizer,
    "bitsandbytes_4bit": Bnb4BitHfQuantizer,
    "bitsandbytes_8bit": Bnb8BitHfQuantizer,
    "gptq": GptqHfQuantizer,
    "aqlm": AqlmHfQuantizer,
    "quanto": QuantoHfQuantizer,
    "eetq": EetqHfQuantizer,
    "hqq": HqqHfQuantizer,
    "fbgemm_fp8": FbgemmFp8HfQuantizer,
    "torchao": TorchAoHfQuantizer,
}

AUTO_QUANTIZATION_CONFIG_MAPPING = {
    "awq": AwqConfig,
    "bitsandbytes_4bit": BitsAndBytesConfig,
    "bitsandbytes_8bit": BitsAndBytesConfig,
    "eetq": EetqConfig,
    "gptq": GPTQConfig,
    "aqlm": AqlmConfig,
    "quanto": QuantoConfig,
    "hqq": HqqConfig,
    "fbgemm_fp8": FbgemmFp8Config,
    "torchao": TorchAoConfig,
}


def _load_entrypoint_quantizers():
    from importlib.metadata import entry_points

    from .auto import HfQuantizerPlugin

    group_name = "hf_quantizers"
    if sys.version_info < (3, 10):
        eps = entry_points()
        eps = eps[group_name] if group_name in eps else []
        eps = {ep.name: ep for ep in eps}
    else:
        eps = entry_points(group=group_name)
        eps = {name: eps[name] for name in eps.names}

    for quantizer_name, quantizer_plugin in eps.items():
        if not isinstance(quantizer_plugin, HfQuantizerPlugin):
            raise ValueError(f"Quantizer Plugin {quantizer_name} does not implement `HfQuantizerPlugin`.")

        AUTO_QUANTIZER_MAPPING[quantizer_name] = quantizer_plugin.get_quantizer()
        AUTO_QUANTIZATION_CONFIG_MAPPING[quantizer_name] = quantizer_plugin.get_config()


def _get_config(quant_method):
    if quant_method not in AUTO_QUANTIZATION_CONFIG_MAPPING:
        _load_entrypoint_quantizers()

    if quant_method not in AUTO_QUANTIZATION_CONFIG_MAPPING.keys():
        raise ValueError(
            f"Unknown quantization type, got {quant_method} - supported types are:"
            f" {list(AUTO_QUANTIZATION_CONFIG_MAPPING.keys())}"
        )

    target_cls = AUTO_QUANTIZATION_CONFIG_MAPPING[quant_method]
    return target_cls


class AutoQuantizationConfig:
    """
    The Auto-HF quantization config class that takes care of automatically dispatching to the correct
    quantization config given a quantization config stored in a dictionary.
    """

    @classmethod
    def from_dict(cls, quantization_config_dict: Dict):
        quant_method = quantization_config_dict.get("quant_method", None)
        # We need a special care for bnb models to make sure everything is BC ..
        if quantization_config_dict.get("load_in_8bit", False) or quantization_config_dict.get("load_in_4bit", False):
            suffix = "_4bit" if quantization_config_dict.get("load_in_4bit", False) else "_8bit"
            quant_method = QuantizationMethod.BITS_AND_BYTES + suffix
        elif quant_method is None:
            raise ValueError(
                "The model's quantization config from the arguments has no `quant_method` attribute. Make sure that the model has been correctly quantized"
            )

        if quant_method not in AUTO_QUANTIZATION_CONFIG_MAPPING.keys():
            raise ValueError(
                f"Unknown quantization type, got {quant_method} - supported types are:"
                f" {list(AUTO_QUANTIZER_MAPPING.keys())}"
            )

        target_cls = _get_config(quant_method)
        return target_cls.from_dict(quantization_config_dict)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        if getattr(model_config, "quantization_config", None) is None:
            raise ValueError(
                f"Did not found a `quantization_config` in {pretrained_model_name_or_path}. Make sure that the model is correctly quantized."
            )
        quantization_config_dict = model_config.quantization_config
        quantization_config = cls.from_dict(quantization_config_dict)
        # Update with potential kwargs that are passed through from_pretrained.
        quantization_config.update(kwargs)
        return quantization_config


class AutoHfQuantizer:
    """
     The Auto-HF quantizer class that takes care of automatically instantiating to the correct
    `HfQuantizer` given the `QuantizationConfig`.
    """

    @classmethod
    def from_config(cls, quantization_config: Union[QuantizationConfigMixin, Dict], **kwargs):
        # Convert it to a QuantizationConfig if the q_config is a dict
        if isinstance(quantization_config, dict):
            quantization_config = AutoQuantizationConfig.from_dict(quantization_config)

        quant_method = quantization_config.quant_method

        # Again, we need a special care for bnb as we have a single quantization config
        # class for both 4-bit and 8-bit quantization
        if quant_method == QuantizationMethod.BITS_AND_BYTES:
            if quantization_config.load_in_8bit:
                quant_method += "_8bit"
            else:
                quant_method += "_4bit"

        if quant_method not in AUTO_QUANTIZER_MAPPING.keys():
            raise ValueError(
                f"Unknown quantization type, got {quant_method} - supported types are:"
                f" {list(AUTO_QUANTIZER_MAPPING.keys())}"
            )

        target_cls = AUTO_QUANTIZER_MAPPING[quant_method]
        return target_cls(quantization_config, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        quantization_config = AutoQuantizationConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls.from_config(quantization_config)

    @classmethod
    def merge_quantization_configs(
        cls,
        quantization_config: Union[dict, QuantizationConfigMixin],
        quantization_config_from_args: Optional[QuantizationConfigMixin],
    ):
        """
        handles situations where both quantization_config from args and quantization_config from model config are present.
        """
        if quantization_config_from_args is not None:
            warning_msg = (
                "You passed `quantization_config` or equivalent parameters to `from_pretrained` but the model you're loading"
                " already has a `quantization_config` attribute. The `quantization_config` from the model will be used."
            )
        else:
            warning_msg = ""

        if isinstance(quantization_config, dict):
            quantization_config = AutoQuantizationConfig.from_dict(quantization_config)

        if (
            isinstance(quantization_config, (GPTQConfig, AwqConfig, FbgemmFp8Config))
            and quantization_config_from_args is not None
        ):
            # special case for GPTQ / AWQ / FbgemmFp8 config collision
            loading_attr_dict = quantization_config_from_args.get_loading_attributes()
            for attr, val in loading_attr_dict.items():
                setattr(quantization_config, attr, val)
            warning_msg += f"However, loading attributes (e.g. {list(loading_attr_dict.keys())}) will be overwritten with the one you passed to `from_pretrained`. The rest will be ignored."

        if warning_msg != "":
            warnings.warn(warning_msg)

        return quantization_config
