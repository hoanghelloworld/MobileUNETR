"""
Utility script to convert PyTorch models to safetensors format
This helps mitigate the vulnerability in torch.load()
"""

import os
import argparse
import torch
from pathlib import Path

def convert_to_safetensors(input_file, output_file=None):
    """Convert a PyTorch model to safetensors format"""
    try:
        from safetensors.torch import save_file
    except ImportError:
        print("Error: safetensors package not installed. Please install it with:")
        print("pip install safetensors")
        return False
    
    try:
        # Load the model
        print(f"Loading model from {input_file}...")
        state_dict = torch.load(input_file, map_location="cpu")
        
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            # If the checkpoint contains a state_dict key, use that
            model_state = state_dict["state_dict"]
        elif isinstance(state_dict, dict):
            # Otherwise use the whole dict if it's already a state dict
            model_state = state_dict
        else:
            print(f"Error: Unknown model format in {input_file}")
            return False
        
        # Determine output filename if not provided
        if output_file is None:
            input_path = Path(input_file)
            output_file = str(input_path.parent / f"{input_path.stem}.safetensors")
        
        # Save using safetensors
        print(f"Saving model to {output_file}...")
        save_file(model_state, output_file)
        
        print(f"Successfully converted {input_file} to {output_file}")
        return True
        
    except Exception as e:
        print(f"Error converting model: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch models to safetensors format")
    parser.add_argument("input", type=str, help="Path to the input PyTorch model file")
    parser.add_argument("--output", type=str, help="Path for the output safetensors file (optional)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist")
        return
    
    convert_to_safetensors(args.input, args.output)

if __name__ == "__main__":
    main()
