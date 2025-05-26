import torch
from diffusers import DDPMPipeline
import os
from tqdm import tqdm

# Load your trained model
checkpoint_path = "diffusion_mri_output_original_unet/models/checkpoint-299"  # Adjust to your latest checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pipeline
pipeline = DDPMPipeline.from_pretrained(checkpoint_path)
pipeline = pipeline.to(device)

# Set output directory
output_dir = "generated_samples_individual"
os.makedirs(output_dir, exist_ok=True)

# Number of images to generate
num_images = 500

# Generate images in batches for efficiency
batch_size = 8  # Adjust based on your GPU memory
num_batches = (num_images + batch_size - 1) // batch_size  # Ceiling division

print(f"Generating {num_images} images...")

image_count = 0
for batch_idx in tqdm(range(num_batches)):
    # Calculate how many images to generate in this batch
    current_batch_size = min(batch_size, num_images - image_count)
    
    # Set different seeds for different batches to ensure variety
    seed = 42 + batch_idx
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # Generate batch of images
    batch_images = pipeline(
        batch_size=current_batch_size,
        generator=generator,
    ).images
    
    # Save each image separately
    for img in batch_images:
        img.save(f"{output_dir}/sample_{image_count:03d}.png")
        image_count += 1
        
        # Break if we've generated enough images
        if image_count >= num_images:
            break

print(f"Successfully generated {image_count} images in '{output_dir}' directory")