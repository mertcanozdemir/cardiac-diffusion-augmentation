import os
import torch
from tqdm.auto import tqdm
from diffusers import DDPMPipeline
from accelerate import Accelerator

from utils import evaluate, load_from_checkpoint, setup_directories

# Main training loop with validation and CUDA support
def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, val_dataloader, lr_scheduler):
    # Initialize accelerator with CUDA
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
        device_placement=True
    )
    
    # Setup directories
    models_dir = setup_directories(config, accelerator)

    # Check for checkpoints to resume from
    latest_checkpoint_dir = None
    if os.path.exists(models_dir):
        checkpoint_dirs = [d for d in os.listdir(models_dir) 
                          if d.startswith("checkpoint-")]
        if checkpoint_dirs:
            # Get the latest checkpoint
            checkpoint_dirs.sort(key=lambda x: int(x.split("-")[1]))
            latest_checkpoint_dir = os.path.join(models_dir, checkpoint_dirs[-1])
            load_from_checkpoint(model, latest_checkpoint_dir)

    # Prepare everything with accelerator
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )

    global_step = 0
    start_epoch = 0
    if latest_checkpoint_dir:
        start_epoch = int(latest_checkpoint_dir.split("-")[-1]) + 1
        print(f"Resuming training from epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, config.num_epochs):
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch["images"]
            
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = torch.nn.functional.mse_loss(noise_pred, noise)
                
                # Backpropagation
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            train_loss += loss.detach().item()
            
            # Update progress bar
            progress_bar.update(1)
            logs = {"train_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # Calculate average training loss
        train_loss /= len(train_dataloader)
        
        # Validation loop
        if val_dataloader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_dataloader:
                    clean_images = batch["images"]
                    noise = torch.randn(clean_images.shape).to(clean_images.device)
                    bs = clean_images.shape[0]
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device).long()
                    noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
                    noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                    loss = torch.nn.functional.mse_loss(noise_pred, noise)
                    val_loss += loss.detach().item()
            
            val_loss /= len(val_dataloader)
            accelerator.log({"val_loss": val_loss}, step=global_step)
            print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        else:
            print(f"Epoch {epoch} | Train Loss: {train_loss:.4f}")

        # After each epoch, optionally generate sample images and save the model
        if accelerator.is_main_process:
            # Create the pipeline
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline, os.path.join(config.output_dir, "samples"))

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                pipeline.save_pretrained(os.path.join(config.output_dir, "models", f"checkpoint-{epoch}"))
                
    return pipeline