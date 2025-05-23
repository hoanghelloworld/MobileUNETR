import os
import cv2
import wandb
import torch
import random
import numpy as np
import warnings

"""
Utils File Used for Training/Validation/Testing
"""


##################################################################################################
def log_metrics(**kwargs) -> None:
    # data to be logged
    log_data = {}
    log_data.update(kwargs)

    # log the data
    wandb.log(log_data)


##################################################################################################
def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar") -> None:
    # print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


##################################################################################################
def safe_load_model(model_path, map_location=None):
    """
    Safely load a model considering PyTorch version and safetensors availability
    """
    # Check PyTorch version
    torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
    using_safe_torch = torch_version >= (2, 6)
    
    # Check if safetensors is available
    try:
        from safetensors.torch import load_file
        has_safetensors = True
    except ImportError:
        has_safetensors = False
        
    # Check if a safetensors version exists
    safetensor_path = model_path.replace('.pth', '.safetensors')
    safetensor_path = safetensor_path.replace('.pt', '.safetensors')
    safetensor_path = safetensor_path.replace('.bin', '.safetensors')
    
    # Try to load with safetensors first if the file exists
    if os.path.exists(safetensor_path) and has_safetensors:
        from safetensors.torch import load_file
        print(f"Loading model using safetensors: {safetensor_path}")
        return load_file(safetensor_path)
    else:
        # Fall back to torch.load with version check
        if not using_safe_torch:
            warnings.warn(
                "You are using torch.load with PyTorch < 2.6.0, which has a known security vulnerability. "
                "Consider upgrading PyTorch to 2.6+ or installing safetensors. "
                "See: https://nvd.nist.gov/vuln/detail/CVE-2025-32434",
                UserWarning
            )
            
        print(f"Loading model using torch.load: {model_path}")
        # Add weights_only=False parameter to address the vulnerability issue
        if using_safe_torch:
            return torch.load(model_path, map_location=map_location, weights_only=False)
        else:
            return torch.load(model_path, map_location=map_location)


##################################################################################################
def load_checkpoint(config, model, optimizer, load_optimizer=True):
    print("=> Loading checkpoint")
    checkpoint = safe_load_model(config.checkpoint_file_name, map_location=config.device)
    model.load_state_dict(checkpoint["state_dict"])

    if load_optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])

        # If we don't do this then it will just have learning rate of old checkpoint
        # and it will lead to many hours of debugging \:
        for param_group in optimizer.param_groups:
            param_group["lr"] = config.learning_rate

    return model, optimizer


##################################################################################################
def seed_everything(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


##################################################################################################
def initialize_weights(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight.data, 1)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0)


##################################################################################################
