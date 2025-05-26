import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup

from config import TrainingConfig, setup_cuda
from dataset import CardiacMRIDataset, get_transforms
from model import create_unet_model
from utils import evaluate
from train import train_loop

def main():
    # Initialize config
    config = TrainingConfig()
    
    # Set up CUDA environment
    setup_cuda(config)
    
    # Set seeds for reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Set CUDA deterministic for reproducibility
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Create transforms
    train_transform = get_transforms(config, train=True)
    val_transform = get_transforms(config, train=False)
    
    # Create dataset
    # Replace with your dataset path
    full_dataset = CardiacMRIDataset(image_dir="C:/Users/Mertcan/Desktop/gata/Real-Images", 
                                     transform=train_transform)
    
    # Split into train and validation sets
    val_size = int(len(full_dataset) * config.validation_split)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(config.seed)
    )
    
    # Update validation set transform
    val_dataset.dataset.transform = val_transform
    
    # Create dataloaders with CUDA pinning
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config.train_batch_size, 
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=config.eval_batch_size, 
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    ) if val_size > 0 else None
    
    # Create UNet model
    model = create_unet_model(config)
    
    # Create noise scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="linear",  # Original linear schedule
        prediction_type="epsilon"
    )
    
    # Create optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Create learning rate scheduler
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
        num_cycles=0.5
    )
    
    # Start training
    pipeline = train_loop(config, model, noise_scheduler, optimizer, train_dataloader, val_dataloader, lr_scheduler)
    
    # Generate a final batch of images
    final_images = evaluate(config, config.num_epochs, pipeline, config.output_dir)
    
    print(f"Training complete. Model saved to {config.output_dir}/models")
    print(f"Sample images saved to {config.output_dir}/samples")
    
    # Display the final image grid
    plt.figure(figsize=(10, 10))
    plt.imshow(final_images, cmap='gray')
    plt.axis('off')
    plt.savefig(f"{config.output_dir}/final_samples.png")
    plt.show()
    
if __name__ == "__main__":
    main()