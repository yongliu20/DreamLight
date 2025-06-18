import numpy as np 
import os, cv2
import torch
from .detail_fixer import DetailsFixer
import folder_paths


class FixDetails:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "structure":("IMAGE",),
                "color":("IMAGE",),
                "model":(["large", "medium"], {"default":"medium"}), 
                "method":(["low_res", "mix_res", "high_res", "wavelet"], {"default":"mix_res"}),
            },
            "optional":{
                "mask":("IMAGE",),
                "shorter_size":("INT", {"default": 512, "min": 256, "max": 1024, "step": 8,}),
            },
        }
    
    CATEGORY = "dreamlight"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "fix_details"

    def __init__(self,):
        self.details_fixer = None#DetailsFixer(model_path="models/Dreamlight/vqmodel")
        self.model = None

    def initialize_details_fixer(self, model_name):
        if model_name == "large":
            self.details_fixer = DetailsFixer(model_path=os.path.join(folder_paths.models_dir, "Dreamlight/vqmodel"), model_size="large")
        elif model_name == "medium":
            self.details_fixer = DetailsFixer(model_path=os.path.join(folder_paths.models_dir, "Dreamlight/vqmodel_medium"), model_size="medium")

    def fix_details(self, structure, color, model="medium", method="mix_res", mask=None, shorter_size=512):
        if self.model is None or self.model != model:
            self.initialize_details_fixer(model)
            self.model = model
        self.details_fixer.shorter_size = shorter_size
        structure = (structure[0].squeeze(-1).detach().data.cpu().numpy()*255.).astype(np.uint8)
        color = (color[0].squeeze(-1).detach().data.cpu().numpy()*255.).astype(np.uint8)
        if mask is None:
            mask = np.ones_like(structure)*255
        else:
            mask = (mask[0].squeeze(-1).detach().data.cpu().numpy()*255.).astype(np.uint8)
        if color.shape[:2] != structure.shape[:2]:
            color = cv2.resize(color, (structure.shape[1], structure.shape[0]))
        
        if method == "low_res":
            func = self.details_fixer.run 
        elif method == "mix_res":
            func = self.details_fixer.run_mixres
        elif method == "high_res":
            func = self.details_fixer.run_hr 
        else:
            func = self.details_fixer.run_wavelet 
        result = func(structure, color, mask)
        result = torch.from_numpy(result.astype(np.float32)/255.).unsqueeze(0)
        return (result,)


NODE_CLASS_MAPPINGS = {
    "fix_details": FixDetails,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "fix_details": "Fix Details",
}