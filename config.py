import os
import torch

# Configuration with CUDA settings
class TrainingConfig:
    def __init__(self):
        self.image_size = 128  # the generated image resolution
        self.train_batch_size = 16
        self.eval_batch_size = 16
        self.num_epochs = 500
        
        self.gradient_accumulation_steps = 4
        self.learning_rate = 2e-4
        self.weight_decay = 1e-5
        self.lr_warmup_steps = 100
        self.save_image_epochs = 5
        self.save_model_epochs = 10
        self.mixed_precision = 'fp16'
        self.output_dir = 'diffusion_mri_output'
        
        self.validation_split = 0.1
        self.overwrite_output_dir = True
        self.seed = 42
        
        # CUDA settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cuda_visible_devices = "0"
        self.num_workers = 4

# Function to setup CUDA environment
def setup_cuda(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices
    print(f"Using device: {config.device}")
    
    if torch.cuda.is_available():
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("CUDA is not available. Using CPU for training.")