#from .modeling_qwen2_vl_dart_vit_ffn import Qwen2VLForConditionalGeneration
#from .modeling_qwen2_vl_dart_vit_ffn2 import Qwen2VLForConditionalGeneration

#from .modeling_qwen2_vl_dart_vit_mean_shift import Qwen2VLForConditionalGeneration
#from .modeling_qwen2_vl_dart_vit_mean_shift_cluster import Qwen2VLForConditionalGeneration
#from .modeling_qwen2_vl_dart_vit_mean_shift_fix import Qwen2VLForConditionalGeneration
#from .modeling_qwen2_vl_dart_vit_mean_shift_similarity import Qwen2VLForConditionalGeneration

#from .modeling_qwen2_vl_dart_vit_mean_shift_cluster_kv import Qwen2VLForConditionalGeneration
#from .modeling_qwen2_vl_dart_vit import Qwen2VLForConditionalGeneration



#from .modeling_qwen2_vl_base import Qwen2VLForConditionalGeneration

#from .modeling_qwen2_vl_IPCV import Qwen2VLForConditionalGeneration

#from .modeling_qwen2_vl_IPCV_FastV import Qwen2VLForConditionalGeneration

#from .modeling_qwen2_vl_IPCV_V2Drop import Qwen2VLForConditionalGeneration

#from .modeling_qwen2_vl_IPCV_SparseVLM import Qwen2VLForConditionalGeneration

#from .modeling_qwen2_vl_IPCV_no_integration import Qwen2VLForConditionalGeneration

#from .modeling_qwen2_vl_ToMe import Qwen2VLForConditionalGeneration

#from .modeling_qwen2_vl_V2Drop import Qwen2VLForConditionalGeneration

#from .modeling_qwen2_vl_FastV import Qwen2VLForConditionalGeneration

#from .modeling_qwen2_vl_SparseVLM import Qwen2VLForConditionalGeneration

#from .modeling_qwen2_vl_Token_fusion import Qwen2VLForConditionalGeneration


#from .modeling_qwen2_vl_Hiprune import Qwen2VLForConditionalGeneration

#from .modeling_qwen2_vl_Saint import Qwen2VLForConditionalGeneration


#from .modeling_qwen2_vl_SiTo import Qwen2VLForConditionalGeneration

#from .modeling_qwen2_vl_FiCoCo import Qwen2VLForConditionalGeneration


import importlib

def get_model_class(method: str):
    if method == "base":
        module = importlib.import_module("Qwen2VL_DART_ViT.modeling_qwen2_vl_base")
    elif method == "ipcv":
        module = importlib.import_module("Qwen2VL_DART_ViT.modeling_qwen2_vl_IPCV")
    elif method == "default":
        module = importlib.import_module("Qwen2VL_DART_ViT.modeling_qwen2_vl")
    elif method == "fastv":
        module = importlib.import_module("Qwen2VL_DART_ViT.modeling_qwen2_vl_FastV")
    elif method == "sparsevlm":
        module = importlib.import_module("Qwen2VL_DART_ViT.modeling_qwen2_vl_SparseVLM")
    elif method == "v2drop":
        module = importlib.import_module("Qwen2VL_DART_ViT.modeling_qwen2_vl_V2Drop")
    elif method == "tome":
        module = importlib.import_module("Qwen2VL_DART_ViT.modeling_qwen2_vl_ToMe")
    elif method == "tofu":
        module = importlib.import_module("Qwen2VL_DART_ViT.modeling_qwen2_vl_Token_fusion")
    elif method == "saint":
        module = importlib.import_module("Qwen2VL_DART_ViT.modeling_qwen2_vl_Saint")
    elif method == "sito":
        module = importlib.import_module("Qwen2VL_DART_ViT.modeling_qwen2_vl_SiTo")
    elif method == "ficoco":
        module = importlib.import_module("Qwen2VL_DART_ViT.modeling_qwen2_vl_FiCoCo")
    elif method == "hiprune":
        module = importlib.import_module("Qwen2VL_DART_ViT.modeling_qwen2_vl_Hiprune")
    else:
        raise ValueError(f"Unknown method: {method}")

    return getattr(module, "Qwen2VLForConditionalGeneration")


from .configuration_qwen2_vl_dart_vit import Qwen2VLConfig, Qwen2VLVisionConfig
#from .image_processor_qwen2_vl_dart_vit import load_pretrained_model