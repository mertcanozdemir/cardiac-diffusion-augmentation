import os
import torch
from PIL import Image
from diffusers import DDPMPipeline

# Function to create image grid for visualization
def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('L', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid

# Fixed evaluation function that doesn't pass device to pipeline
def evaluate(config, epoch, pipeline, output_dir):
    # Move pipeline to the right device if needed
    if hasattr(pipeline, "to") and hasattr(config, "device"):
        pipeline = pipeline.to(config.device)
    
    # Sample some images from random noise (backwards diffusion process)
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.manual_seed(config.seed),
    ).images

    # Make a grid out of the images
    image_grid = make_grid(images, rows=2, cols=2)

    # Save the images
    os.makedirs(output_dir, exist_ok=True)
    image_grid.save(f"{output_dir}/sample_{epoch:04d}.png")
    
    return image_grid

# Function to load from checkpoint
def load_from_checkpoint(model, checkpoint_path):
    if os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        pipeline = DDPMPipeline.from_pretrained(checkpoint_path)
        model.load_state_dict(pipeline.unet.state_dict())
        return True
    return False

# Function to set up directory structure
def setup_directories(config, accelerator=None):
    if accelerator is None or accelerator.is_main_process:
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(os.path.join(config.output_dir, "samples"), exist_ok=True)
        os.makedirs(os.path.join(config.output_dir, "models"), exist_ok=True)
        
        if accelerator is not None:
            accelerator.init_trackers("cardiac_mri_diffusion")
    
    return os.path.join(config.output_dir, "models")