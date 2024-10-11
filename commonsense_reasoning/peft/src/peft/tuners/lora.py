# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
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
import importlib
import math
import re
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

####################  debug  ####################
# import sys 
# sys.path.append('peft/src/peft/')
# from utils import PeftConfig, PeftType, transpose
####################  debug  ####################

from ..utils import PeftConfig, PeftType, transpose
from einops import rearrange


def is_bnb_available():
    return importlib.util.find_spec("bitsandbytes") is not None


if is_bnb_available():
    import bitsandbytes as bnb


@dataclass
class LoraConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.Lora`].

    Args:
        r (`int`): Lora attention dimension
        target_modules (`Union[List[str],str]`): The names of the modules to apply Lora to.
        lora_alpha (`float`): The alpha parameter for Lora scaling.
        lora_dropout (`float`): The dropout probability for Lora layers.
        merge_weights (`bool`):
            Whether to merge the weights of the Lora layers with the base transformer model in `eval` mode.
        fan_in_fan_out (`bool`): Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        enable_lora ( `List[bool]`): Used with `lora.MergedLinear`.
        bias (`str`): Bias type for Lora. Can be 'none', 'all' or 'lora_only'
        modules_to_save (`List[str]`):List of modules apart from LoRA layers to be set as trainable
            and saved in the final checkpoint.
    """

    r: int = field(default=8, metadata={"help": "Lora attention dimension"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    lora_alpha: int = field(default=None, metadata={"help": "Lora alpha"})
    lora_dropout: float = field(default=None, metadata={"help": "Lora dropout"})
    merge_weights: bool = field(
        default=False, metadata={"help": "Merge weights of the original model and the Lora model"}
    )
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    enable_lora: Optional[List[bool]] = field(default=None, metadata={"help": "Used with `lora.MergedLinear`."})
    bias: str = field(default="none", metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"})
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )
    # expert_A: int = field(default=1, metadata={"help": "expert for lora A"})
    # expert_B: int = field(default=1, metadata={"help": "expert for lora B"})
    lora_router: bool=field(default=False, metadata={"help": "whether to use router"})
    lora_router_mixer: bool=field(default=False, metadata={"help": "whether to use router mixer"})

    def __post_init__(self):
        self.peft_type = PeftType.LORA


class LoraModel(torch.nn.Module):
    """
    Creates Low Rank Adapter (Lora) model from a pretrained transformers model.

    Args:
        model ([`transformers.PreTrainedModel`]): The model to be adapted.
        config ([`LoraConfig`]): The configuration of the Lora model.

    Returns:
        `torch.nn.Module`: The Lora model.

    Example::

        >>> from transformers import AutoModelForSeq2SeqLM, LoraConfig >>> from peft import LoraModel, LoraConfig >>>
        config = LoraConfig(
            peft_type="LORA", task_type="SEQ_2_SEQ_LM", r=8, lora_alpha=32, target_modules=["q", "v"],
            lora_dropout=0.01, )
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base") >>> lora_model = LoraModel(config, model)

    **Attributes**:
        - **model** ([`transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`LoraConfig`]): The configuration of the Lora model.
    """

    def __init__(self, config, model):
        super().__init__()
        self.peft_config = config
        self.model = model
        self._find_and_replace()
        mark_only_lora_as_trainable(self.model, self.peft_config.bias)
        self.forward = self.model.forward

    def _find_and_replace(self):
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if loaded_in_8bit and not is_bnb_available():
            raise ImportError(
                "To use Lora with 8-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )
        is_target_modules_in_base_model = False
        is_hf_device_map_available = hasattr(self.model, "hf_device_map")
        kwargs = {
            "r": self.peft_config.r,
            "lora_alpha": self.peft_config.lora_alpha,
            "lora_dropout": self.peft_config.lora_dropout,
            "fan_in_fan_out": self.peft_config.fan_in_fan_out,
            "merge_weights": (self.peft_config.merge_weights or self.peft_config.inference_mode)
            and not is_hf_device_map_available,
            # "expert_A": self.peft_config.expert_A,
            # "expert_B": self.peft_config.expert_B,
            "lora_router": self.peft_config.lora_router,
            "lora_router_mixer": self.peft_config.lora_router_mixer,
        }
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            if isinstance(self.peft_config.target_modules, str):
                target_module_found = re.fullmatch(self.peft_config.target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in self.peft_config.target_modules)
            if target_module_found:
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = self._get_submodules(key)
                bias = target.bias is not None
                if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
                    kwargs.update(
                        {
                            "has_fp16_weights": target.state.has_fp16_weights,
                            "memory_efficient_backward": target.state.memory_efficient_backward,
                            "threshold": target.state.threshold,
                            "index": target.index,
                        }
                    )
                    if self.peft_config.enable_lora is None:
                        new_module = Linear8bitLt(target.in_features, target.out_features, bias=bias, **kwargs)
                    else:
                        kwargs.update({"enable_lora": self.peft_config.enable_lora})
                        new_module = MergedLinear8bitLt(target.in_features, target.out_features, bias=bias, **kwargs)
                elif isinstance(target, torch.nn.Linear) and self.peft_config.enable_lora is None:
                    new_module = Linear(target.in_features, target.out_features, bias=bias, **kwargs)
                elif self.peft_config.enable_lora is not None:
                    kwargs.update({"enable_lora": self.peft_config.enable_lora})
                    if isinstance(target, Conv1D):
                        in_features, out_features = (
                            target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
                        )
                    else:
                        in_features, out_features = target.in_features, target.out_features
                        if kwargs["fan_in_fan_out"]:
                            warnings.warn(
                                "fan_in_fan_out is set to True but the target module is not a Conv1D. "
                                "Setting fan_in_fan_out to False."
                            )
                            kwargs["fan_in_fan_out"] = self.peft_config.fan_in_fan_out = False
                    new_module = MergedLinear(in_features, out_features, bias=bias, **kwargs)
                self._replace_module(parent, target_name, new_module, target)
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {self.peft_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _get_submodules(self, key):
        parent = self.model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = self.model.get_submodule(key)
        return parent, target, target_name

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        if old_module.bias is not None:
            new_module.bias = old_module.bias
        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "lora_" in name:
                module.to(old_module.weight.device)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    @property
    def modules_to_save(self):
        return None

    def get_peft_config_as_dict(self, inference: bool = False):
        config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(self.peft_config).items()}
        if inference:
            config["inference_mode"] = True
        return config

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                module.disable_adapters = False if enabled else True

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)


# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


# had to adapt it for `lora_only` to work
def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoraLayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


class LoraLayer:
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
        self.disable_adapters = False


class Linear(nn.Linear, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        # expert_A: int = 1,
        # expert_B: int = 1,
        lora_router: bool=False,
        lora_router_mixer: bool=False,
        **kwargs,
    ):  
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        self.lora_router = lora_router
        self.lora_router_mixer = lora_router_mixer
        # Actual trainable parameters
        if r > 0:
            # self.lora_A = nn.ModuleList([nn.Linear(in_features, r, bias=False) for _ in range(expert_A)])
            # self.lora_B = nn.ModuleList([nn.Linear(r, out_features, bias=False) for _ in range(expert_B)])
            
            # we set hyper-parameter here: lora_head = 4
            lora_head = 4
            self.lora_A = nn.Linear(in_features//lora_head, r, bias=False)
            if self.lora_router:
                if self.lora_router_mixer:
                    self.lora_R = nn.Linear(in_features//lora_head, r**2, bias=False)
                else:
                    self.lora_R = nn.Linear(in_features//lora_head, r, bias=False)
            self.lora_B = nn.Linear(r, out_features//lora_head, bias=False)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            # self.scale_a = 1./math.sqrt(expert_A)
            # self.scale_b = 1./math.sqrt(expert_B)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            # for idx in range(len(self.lora_A)):
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            # for idx in range(len(self.lora_B)):
            if self.lora_router:
                nn.init.kaiming_uniform_(self.lora_R.weight, a=math.sqrt(5))
            # nn.init.kaiming_uniform_(self.lora_B.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)
    
    def forward_A(self, x):
        # res = [l(x) for l in self.lora_A]
        # return self.scale_a*torch.sum(torch.stack(res, dim=-1).to(device=x.device), dim=-1, keepdim=False)
        return self.lora_A(x)
    def forward_B(self, x):
        # res = [l(x) for l in self.lora_B]
        # return self.scale_b*torch.sum(torch.stack(res, dim=-1).to(device=x.device), dim=-1, keepdim=False)
        return self.lora_B(x)
    
    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        # for idx in range(len(self.lora_A)):
        self.lora_A.train(mode)
        # for idx in range(len(self.lora_B)):
        if self.lora_router:
            self.lora_R.train(mode)
        self.lora_B.train(mode)

        self.merged = False

    def eval(self):
        nn.Linear.eval(self)
        # for idx in range(len(self.lora_A)):
        self.lora_A.eval()
        # for idx in range(len(self.lora_B)):
        if self.lora_router:
            self.lora_R.eval()
        self.lora_B.eval()

    def forward(self, x: torch.Tensor):
        previous_dtype = self.weight.dtype

        if self.disable_adapters:
            raise NotImplementedError("Disabling adapters is not supported for LoraLayer yet")
        
        dropout_x = self.lora_dropout(x).to(self.lora_A.weight.dtype)
        # we set hyper-parameter here: lora_head = 4
        lora_head = 4
        dropout_x_shape = dropout_x.shape 
        if len(dropout_x_shape) == 3:
            dropout_x = rearrange(dropout_x, 'b s (h d) -> (b h) s d', h=lora_head)
        else:
            dropout_x = rearrange(dropout_x, 'b (h d) -> (b h) d', h=lora_head)

        left_result = self.forward_A(dropout_x)
        router_weight = self.lora_R(dropout_x)
        if self.lora_router:
            router_weight = torch.softmax(router_weight, dim=-1)

            if not self.lora_router_mixer:
                router_weight = torch.diag_embed(router_weight)
            else:
                router_weight_shape = router_weight.shape[:-1] + (self.r, self.r)
                router_weight = router_weight.reshape(router_weight_shape)

        if len(router_weight.shape) == 3:
            moe_result = torch.enisum('br, brk->bk', left_result, router_weight)
        else:
            moe_result = torch.einsum('bnr, bnrk->bnk', left_result, router_weight)

        lora_result = self.forward_B(moe_result) * self.scaling
        
        if len(dropout_x_shape) == 3:
            lora_result = rearrange(lora_result, '(b h) s d -> b s (h d)', h=lora_head)
        else:
            lora_result = rearrange(lora_result, '(b h) d -> b (h d)', h=lora_head)
            
        org_result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        result = lora_result + org_result

        if result.dtype != previous_dtype:
            result = result.to(previous_dtype)

        return result


class MergedLinear(nn.Linear, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs,
    ):
        raise NotImplementedError("MergedLinear is not implemented yet")

if is_bnb_available():

    class Linear8bitLt(bnb.nn.Linear8bitLt, LoraLayer):
        # Lora implemented in a dense layer
        def __init__(
            self,
            in_features,
            out_features,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            **kwargs,
        ):
            raise NotImplementedError

    class MergedLinear8bitLt(bnb.nn.Linear8bitLt, LoraLayer):
        # Lora implemented in a dense layer
        def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            enable_lora: List[bool] = [False],
            **kwargs,
        ):
            raise NotImplementedError
        


if __name__ == '__main__':
    lora_linear = Linear(
        in_features=768,
        out_features=768,
        r=8,
        lora_alpha=16,
        lora_dropout=0.01,
        fan_in_fan_out=False,
        merge_weights=False,
        lora_router=True,
    )

    input_tensor = torch.randn(4, 16, 768)
    output_tensor = lora_linear(input_tensor)
    print(output_tensor.shape)
