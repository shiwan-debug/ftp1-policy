#######################################################################

# This file is copied and updated from the official code of T3 Tactile Encoder. For more details, please refer to the official paper & code.

# Transferable Tactile Transformers for Representation Learning Across Diverse Sensors and Tasks

# Authors: Jialiang (Alan) Zhao (alanzhao@csail.mit.edu), Yuxiang Ma, Lirui Wang, Edward H. Adelson

# Website: https://t3.alanz.info/

# Code: https://github.com/alanzjl/t3

# Updated by: Michael (Chengbo) Yuan (ycb24@mails.tsinghua.edu.cn)

#######################################################################

import os
import torch
from torch import nn
import torchvision
import torch
import torch.nn as nn
from typing import Literal
import pathlib

import timm.models.vision_transformer as timm_vit
from functools import partial
from openpi.shared import download
from openpi.models_pytorch.ftp1_model_config import (
    T3_PRETRAINED_SENSOR_NAME_MAP,
    T3_PRETRAINED_TACTILE_ENCODER_CHECKPOINTS_BASE_URL
)



class T3TactileEncoderConfig:
    
    patch_size: int = 16
    encoder_embed_dim: int = 768
    encoder_heads: int = 12
    encoder_depth: int = 3
    mlp_ratio: int = 4
    
    
class T3SharedChunkConfig:
    
    embed_dim: int = 768
    depth: int = 9
    num_heads: int = 12
    mlp_ratio: int = 4
    pooling_type: str = "none"
    
    
def makeMLP(input_dim,
            output_dim,
            hidden_dims,
            dropout_p,
            tanh_end,
            ln):
    layers = [nn.Linear(input_dim, hidden_dims[0]), nn.SiLU()]
    for i in range(1, len(hidden_dims)):
        layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
        if dropout_p > 1e-5:
            layers.append(nn.Dropout(dropout_p))
        if ln:
            layers.append(nn.LayerNorm(hidden_dims[i]))
        layers.append(nn.SiLU())
    layers.append(nn.Linear(hidden_dims[-1], output_dim))
    if tanh_end:
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)
    
    
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, device='cuda'):
        kwargs = {}
        if not torch.cuda.is_available():
            kwargs['map_location'] = device
        if os.path.exists(path):
            print(f"Loading encoder from weights from {path}", True, "green")
            self.load_state_dict(torch.load(path, **kwargs))
        else:
            # try to finetune from gs_green if it exists
            gs_green_path = path[:path.rfind('/')] + '/gs_green.pth'
            if os.path.exists(gs_green_path):
                print(f"Encoder weights not found at {path}. Loading from gs_green", True, "warning")
                self.load_state_dict(torch.load(gs_green_path, **kwargs))
            else: # if gs_green also doesn't exist, use random initialization
                print(f"Encoder weights not found at {path}. Skipping", True, "warning")


class IdentityEncoder(Encoder):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class ViTEncoder(timm_vit.VisionTransformer, Encoder):
    def __init__(self,
                 tokenizer_name,
                 sensor_name,
                 embed_dim,
                 num_heads,
                 mlp_ratio,
                 depth,
                 norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
                 load_t3_pretrained_checkpoint=False,
                 cache_t3_pretrained_checkpoint_dir=None,
                 **kwargs):
        super(ViTEncoder, self).__init__(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            mlp_ratio=mlp_ratio, 
            depth=depth,
            norm_layer=norm_layer,
            **kwargs)
        
        self.tokenizer_name = tokenizer_name
        self.sensor_name = sensor_name
        self.blocks = nn.ModuleList([
            timm_vit.Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        
        del self.head # remove the head
        del self.norm # remove the normalization at the end, which will be added in the trunk
        
        if load_t3_pretrained_checkpoint:
            self._load_t3_pretrained_checkpoint(cache_t3_pretrained_checkpoint_dir)

    def forward(self, x):     # (B, C, H, W)
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        return x

    def _load_t3_pretrained_checkpoint(self, cache_dir: str = None):
        """
        Load T3 pretrained checkpoint for ViT encoder.
        If checkpoint doesn't exist in cache, download from HuggingFace.
        """
        if cache_dir is None:
            cache_dir = pathlib.Path("./cache")
        else:
            cache_dir = pathlib.Path(cache_dir)
        
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Use sensor_name directly (passed from TactileDataEncoder)
        sensor_name = self.sensor_name
        
        # Map sensor name to T3 checkpoint name
        t3_checkpoint_name = T3_PRETRAINED_SENSOR_NAME_MAP.get(sensor_name, None)
        if t3_checkpoint_name is None:
            print(f"Warning: Sensor name '{sensor_name}' not found in T3_PRETRAINED_SENSOR_NAME_MAP. Skipping pretrained checkpoint loading.")
            return
        
        # Build checkpoint path
        checkpoint_filename = f"{t3_checkpoint_name}.pth"
        checkpoint_path = cache_dir / checkpoint_filename
        
        # Build URL
        checkpoint_url = T3_PRETRAINED_TACTILE_ENCODER_CHECKPOINTS_BASE_URL + checkpoint_filename
        
        # Download if not exists
        if not checkpoint_path.exists():
            print(f"Downloading T3 pretrained checkpoint from {checkpoint_url} to {checkpoint_path}")
            try:
                downloaded_path = download.maybe_download(checkpoint_url)
                # Move to cache_dir if downloaded to different location
                if downloaded_path != checkpoint_path:
                    import shutil
                    shutil.move(str(downloaded_path), str(checkpoint_path))
            except Exception as e:
                print(f"Failed to download T3 pretrained checkpoint: {e}. Skipping.")
                return
        
        # Load checkpoint
        if checkpoint_path.exists():
            self.load(str(checkpoint_path))
        else:
            raise FileNotFoundError(f"T3 pretrained checkpoint not found at {checkpoint_path}. Skipping.")

    def load(self, path):
        """
        Positional embedding interpolation from DeiT
        https://github.com/facebookresearch/deit
        """
        if os.path.exists(path):
            print(f"Loading encoder from weights from {path}. Will apply pos_embed interpolation.", True, "green")
            checkpoint = torch.load(path, map_location='cpu')
        else:
            gs_green_path = path[:path.rfind('/')] + '/gs_green.pth'
            checkpoint = torch.load(gs_green_path, map_location='cpu')
            print(f"Encoder weights not found at {path}. Loading from gs_green", True, "warning")
        if 'pos_embed' in checkpoint:
            pos_embed_checkpoint = checkpoint['pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1]
            num_patches = self.patch_embed.num_patches
            num_extra_tokens = self.pos_embed.shape[-2] - num_patches
            # height (== width) for the checkpoint position embedding
            orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
            # height (== width) for the new position embedding
            new_size = int(num_patches ** 0.5)
            # class_token and dist_token are kept unchanged
            if orig_size != new_size:
                print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                # only the position tokens are interpolated
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                checkpoint['pos_embed'] = new_pos_embed
        self.load_state_dict(checkpoint)
        
        

class Trunk(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path, device='cuda'):
        kwargs = {}
        if not torch.cuda.is_available():
            kwargs['map_location'] = device
        if os.path.exists(path):
            print(f"Loading trunk from weights from {path}", True, "green")
            self.load_state_dict(torch.load(path, **kwargs))
        else:
            print(f"Trunk weights not found at {path}. Skipping", True, "warning")


class IdentityTrunk(Trunk):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x):
        return x

class MLPTrunk(Trunk):
    def __init__(self, 
                 input_dim,
                 output_dim,
                 hidden_dims,
                 dropout_p=0.1,
                 tanh_end=False,
                 ln=False,
                 **kwargs):
        super().__init__()

        self.model = makeMLP(input_dim, output_dim, hidden_dims, dropout_p, tanh_end, ln)
        
    def forward(self, x):
        return self.model(x)


class TransformerTrunk(Trunk):
    """ 
    Transformer with only intermediate blocks and a final normalization layer
    """
    def __init__(self, embed_dim=768, depth=9, num_heads=12,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 pooling_type: Literal['none', 'global', 'cls'] = 'none',
                 load_t3_pretrained_checkpoint=False,
                 cache_t3_pretrained_checkpoint_dir=None,
                 tokenizer_name: str = None,
                 sensor_name: str = None,
                 **kwargs):
        super().__init__()
        self.tokenizer_name = tokenizer_name
        self.sensor_name = sensor_name

        self.blocks = nn.ModuleList([
            timm_vit.Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.pooling_type = pooling_type

        self.apply(self._init_weights)
        if load_t3_pretrained_checkpoint:
            self._load_t3_pretrained_checkpoint(cache_t3_pretrained_checkpoint_dir)

    def _load_t3_pretrained_checkpoint(self, cache_dir: str = None):
        """
        Load T3 pretrained checkpoint for TransformerTrunk (shared chunk encoder).
        The checkpoint URL is fixed: base_url/trunk.pth
        If checkpoint doesn't exist in cache, download from HuggingFace.
        """
        if cache_dir is None:
            cache_dir = pathlib.Path("./cache")
        else:
            cache_dir = pathlib.Path(cache_dir)
        
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Fixed checkpoint filename for trunk
        checkpoint_filename = "trunk.pth"
        checkpoint_path = cache_dir / checkpoint_filename
        
        # Fixed URL for trunk checkpoint
        checkpoint_url = T3_PRETRAINED_TACTILE_ENCODER_CHECKPOINTS_BASE_URL + "trunk.pth"
        
        # Download if not exists
        if not checkpoint_path.exists():
            print(f"Downloading T3 pretrained trunk checkpoint from {checkpoint_url} to {checkpoint_path}")
            try:
                downloaded_path = download.maybe_download(checkpoint_url)
                # Move to cache_dir if downloaded to different location
                if downloaded_path != checkpoint_path:
                    import shutil
                    shutil.move(str(downloaded_path), str(checkpoint_path))
            except Exception as e:
                print(f"Failed to download T3 pretrained trunk checkpoint: {e}. Skipping.")
                return
        
        # Load checkpoint
        if checkpoint_path.exists():
            self.load(str(checkpoint_path))
        else:
            raise FileNotFoundError(f"T3 pretrained trunk checkpoint not found at {checkpoint_path}.")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        is_mae = False
        if isinstance(x, tuple):
            (x, mask, ids_restore) = x
            is_mae = True
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        if self.pooling_type == 'none':
            x = self.norm(x)
        elif self.pooling_type == 'global':
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            #TODO: maybe add another norm layer here
        elif self.pooling_type == 'cls':
            x = self.norm(x)
            x = x[:, 0]

        if is_mae:
            return (x, mask, ids_restore)
        else: 
            return x