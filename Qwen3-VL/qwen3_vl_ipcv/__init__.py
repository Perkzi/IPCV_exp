# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
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
from typing import TYPE_CHECKING

from transformers.utils import _LazyModule
from transformers.utils.import_utils import define_import_structure


# if TYPE_CHECKING:
#     from .configuration_qwen3_vl import *
#     #from .modeling_qwen3_vl import *
#     from .modeling_qwen3_vl_base import *
#     from .processing_qwen3_vl import *
#     from .video_processing_qwen3_vl import *
# else:
#     # import sys

#     # _file = globals()["__file__"]
#     # sys.modules[__name__] = _LazyModule(__name__, _file, define_import_structure(_file), module_spec=__spec__)

    
from .configuration_qwen3_vl import *

from .processing_qwen3_vl import *
from .video_processing_qwen3_vl import *



import importlib

def get_model_class(method: str):
    if method == "base":
        module = importlib.import_module("qwen3_vl_ipcv.modeling_qwen3_vl_base")
    elif method == "ipcv":
        module = importlib.import_module("qwen3_vl_ipcv.modeling_qwen3_vl_IPCV")
    elif method == "default":
        module = importlib.import_module("qwen3_vl_ipcv.modeling_qwen3_vl")
    elif method == "fastv":
        module = importlib.import_module("qwen3_vl_ipcv.modeling_qwen3_vl_FastV")
    elif method == "sparsevlm":
        module = importlib.import_module("qwen3_vl_ipcv.modeling_qwen3_vl_SparseVLM")
    elif method == "v2drop":
        module = importlib.import_module("qwen3_vl_ipcv.modeling_qwen3_vl_V2Drop")
    else:
        raise ValueError(f"Unknown method: {method}")

    return getattr(module, "Qwen3VLForConditionalGeneration")

#from .modeling_qwen3_vl import *
# from .modeling_qwen3_vl_base import *
#from .modeling_qwen3_vl_IPCV import *