from diffusers import UNet2DModel

# Function to create the UNet model
def create_unet_model(config):
    # Create ORIGINAL UNet model with default parameters
    model = UNet2DModel(
        sample_size=config.image_size,
        in_channels=1,
        out_channels=1,
        center_input_sample=False,
        time_embedding_type="positional",
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D", 
            "DownBlock2D",
            "AttnDownBlock2D",  # Added attention here
            "AttnDownBlock2D",  # Added attention here
        ),
        # Modified upsampling path with attention for better feature reconstruction
        up_block_types=(
            "AttnUpBlock2D",  # Added attention here
            "AttnUpBlock2D",  # Added attention here
            "UpBlock2D", 
            "UpBlock2D", 
            "UpBlock2D",
        ),
        block_out_channels=(64, 128, 256, 512, 1024),
        layers_per_block=2,
        downsample_padding=0,
        # Change activation function to silu which is the default
        act_fn="silu",
        norm_num_groups=16,
        add_attention=True,
        mid_block_scale_factor=1.0,
        # Try changing this to match your version
        resnet_time_scale_shift="default"
    )
    
    return model



