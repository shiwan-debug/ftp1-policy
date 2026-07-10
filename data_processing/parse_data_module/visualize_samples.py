import argparse
import sys
from pathlib import Path
import numpy as np
import cv2

# Add path for ReplayBuffer
_SCRIPT_DIR = Path(__file__).resolve().parent
_DATA_PROCESSING_DIR = _SCRIPT_DIR.parent
if str(_DATA_PROCESSING_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_PROCESSING_DIR))

from common.replay_buffer import ReplayBuffer

def save_image(img_data, output_path):
    """Save the middle frame of the image sequence."""
    if img_data.ndim == 4: # (T, H, W, C)
        # Take middle frame to ensure content
        idx = img_data.shape[0] // 2
        img = img_data[idx]
    elif img_data.ndim == 3: # (H, W, C)
        img = img_data
    else:
        print(f"Skipping image with unexpected shape {img_data.shape}")
        return

    # Check for empty or invalid images
    if img.size == 0 or np.max(img) == 0:
        print(f"[WARN] Image empty or all zeros at {output_path}")
        
    # Convert RGB to BGR for OpenCV
    if img.shape[-1] == 3:
        img = img[..., ::-1]
    
    try:
        cv2.imwrite(str(output_path), img)
        print(f"[OK] Saved {output_path}")
    except Exception as e:
        print(f"[ERROR] Failed writing {output_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Extract sample images from a Zarr file.")
    parser.add_argument("--zarr_path", type=str, required=True, help="Path to the Zarr file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save images.")
    parser.add_argument("--dataset_name", type=str, default="dataset", help="Dataset name prefix.")
    args = parser.parse_args()

    zarr_path = Path(args.zarr_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not zarr_path.exists():
        print(f"[ERROR] Zarr path not found: {zarr_path}")
        return

    try:
        # Open in read-only mode
        rb = ReplayBuffer.create_from_path(str(zarr_path), mode="r")
        
        # Keys to look for based on provided READMEs
        keys_to_check = [
            "camera_ego_rgb",
            "right_wrist_camera_rgb",
            "left_wrist_camera_rgb",
             # Fallback/Other possible keys
            "camera_rgb", 
            "wrist_camera_rgb"
        ]

        found_any = False
        for key in keys_to_check:
            if key in rb.data:
                try:
                    # Access data
                    data_arr = rb.data[key]
                    # Convert to numpy and save
                    img_data = np.array(data_arr)
                    
                    save_name = f"{args.dataset_name}_{zarr_path.stem}_{key}.jpg"
                    save_image(img_data, output_dir / save_name)
                    found_any = True
                except Exception as e:
                    print(f"[ERROR] Failed processing key {key}: {e}")
        
        if not found_any:
            print(f"[WARN] No camera keys ({keys_to_check}) found in {zarr_path.name}")

    except Exception as e:
        print(f"[ERROR] Error reading {zarr_path}: {e}")

if __name__ == "__main__":
    main()
