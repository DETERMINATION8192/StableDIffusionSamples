from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch

model_id = "KBlueLeaf/kohaku-v2.1"

pipeline = StableDiffusionPipeline.from_pretrained(model_id)
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
pipeline.to("cuda")

images = pipeline("girl", num_inference_steps=20, torch_dtype=torch.float16).images
images[0].save("image.png")