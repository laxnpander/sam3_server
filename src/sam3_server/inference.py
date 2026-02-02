import torch
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
#from sam3.visualization_utils import plot_results
import numpy as np

class SAM3Inference:
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = torch.device(device)

        self.model = build_sam3_image_model(
            checkpoint_path=checkpoint_path
        )
        self.model.to(device=self.device)
        self.processor = processor = Sam3Processor(self.model)

    def run_inference(self, image, prompt: str):

        inference_state = self.processor.set_image(image)
        output = self.processor.set_text_prompt(state=inference_state, prompt=prompt)

        #plot_results(image, inference_state)

        masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

        masks_np = self.tensor_to_numpy(masks)
        boxes_np = boxes.cpu().numpy().tolist()
        scores_np = scores.cpu().numpy().tolist()

        return {
            "masks": masks_np,
            "boxes": boxes_np,
            "scores": scores_np
        }

    def tensor_to_numpy(self, masks_tensor):
        if masks_tensor is None or masks_tensor.numel() == 0:
            return []

        masks_np = masks_tensor.cpu().numpy()

        b, n, h, w = masks_np.shape
        flattened_masks = masks_np.reshape(-1, h, w)

        np_mats = []
        for i in range(flattened_masks.shape[0]):
            mask = (flattened_masks[i] > 0).astype(np.uint8) * 255
            np_mats.append(mask)

        return np_mats