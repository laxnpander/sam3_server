import torch
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
import numpy as np
import cv2

class SAM3Inference:
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = torch.device(device)

        self.model = build_sam3_image_model(
            checkpoint_path=checkpoint_path
        )
        self.model.to(device=self.device)
        self.processor = processor = Sam3Processor(self.model)

    def run_inference(self, image_bytes, prompt: str):
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        inference_state = self.processor.set_image(image)

        output = self.processor.set_text_prompt(state=inference_state, prompt=prompt)
        masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

        mask_np = masks.squeeze().cpu().numpy().astype(np.uint8) * 255
        boxes_np = boxes.cpu().numpy().tolist()
        scores_np = scores.cpu().numpy().tolist()

        return {
            "mask": [],
            "boxes": boxes_np,
            "scores": scores_np
        }