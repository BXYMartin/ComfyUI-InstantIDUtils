from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
import numpy as np
from PIL import Image
import torch


class AnyType(str):
    """A special class that is always equal in not equal comparisons. Credit to pythongosssss"""

    def __eq__(self, _) -> bool:
        return True

    def __ne__(self, __value: object) -> bool:
        return False

any = AnyType("*")


class MultiControlNetConverter:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "controlnets": (any,),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("multicontrolnet",)
    CATEGORY = "ControlNetUtils"
    FUNCTION = "execute"

    def execute(self, controlnets):
        model = MultiControlNetModel(controlnets)
        return [model]
    
    
class NHWC2NCHWTensor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
            }
        }

    CATEGORY = "ControlNetUtils"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    FUNCTION = "execute"
    def execute(self, image):
        print(image.shape)
        image.permute(0, 3, 1, 2)
        print(image.shape)
        return [image]
    
class NHWCTensor2PIL:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
            }
        }

    CATEGORY = "ControlNetUtils"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    FUNCTION = "execute"
    
    def execute(self, image):
        # convert NHWC Tensor to PIL Image
        img_array = image.squeeze(0).cpu().numpy() * 255.0
        img_pil = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
        return [img_pil]
    
    
class PIL2NHWCTensor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
            }
        }

    CATEGORY = "ControlNetUtils"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    FUNCTION = "execute"
    def execute(self, image):
        # 将 PIL.Image 转换为 numpy.ndarray
        img_array = np.array(image)
        # 转换 numpy.ndarray 为 torch.Tensor
        img_tensor = torch.from_numpy(img_array).float() / 255.
        # 转换图像格式为 CHW (如果需要)
        if img_tensor.ndim == 3 and img_tensor.shape[-1] == 3:
            img_tensor = img_tensor.permute(2, 0, 1)
        # 添加批次维度并转换为 NHWC
        img_tensor = img_tensor.unsqueeze(0).permute(0, 2, 3, 1)
        return [img_tensor]
    
    
class ListOfImages:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
            },
            "optional": {
                "any_1": ("IMAGE",),
                "any_2": ("IMAGE",),
                "any_3": ("IMAGE",),
                "any_4": ("IMAGE",),
                "any_5": ("IMAGE",),
                "any_6": ("IMAGE",),
                "any_7": ("IMAGE",),
                "any_8": ("IMAGE",),
            }
        }

    CATEGORY = "ControlNetUtils"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)

    FUNCTION = "execute"

    def execute(self,
                any_1=None,
                any_2=None,
                any_3=None,
                any_4=None,
                any_5=None,
                any_6=None,
                any_7=None,
                any_8=None):

        list_any = []

        if any_1 is not None:
            try:
                list_any.append(any_1)
            except Exception as e:
                raise e
        if any_2 is not None:
            try:
                list_any.append(any_2)
            except Exception as e:
                raise e
        if any_3 is not None:
            try:
                list_any.append(any_3)
            except Exception as e:
                raise e
        if any_4 is not None:
            try:
                list_any.append(any_4)
            except Exception as e:
                raise e
        if any_5 is not None:
            try:
                list_any.append(any_5)
            except Exception as e:
                raise e
        if any_6 is not None:
            try:
                list_any.append(any_6)
            except Exception as e:
                raise e
        if any_7 is not None:
            try:
                list_any.append(any_7)
            except Exception as e:
                raise e
        if any_8 is not None:
            try:
                list_any.append(any_8)
            except Exception as e:
                raise e

        # yes, double brackets are needed because of the OUTPUT_IS_LIST... ¯\_(ツ)_/¯
        return [list_any]

NODE_CLASS_MAPPINGS = {
    "MultiControlNetConverter": MultiControlNetConverter,
    "ListOfImages": ListOfImages,
    "PIL2NHWCTensor": PIL2NHWCTensor,
    "NHWC2NCHWTensor": NHWC2NCHWTensor,
    "NHWCTensor2PIL": NHWCTensor2PIL
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MultiControlNetConverter": "Multi-ControlNet Converter",
    "ListOfImages": "List of Images",
    "PIL2NHWCTensor": "Convert PIL to Tensor (NHWC)",
    "NHWC2NCHWTensor": "Convert Tensor (NHWC) to (NCHW)",
    "NHWCTensor2PIL": "Convert Tensor (NHWC) to PIL"
}