# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 09:22:16 2022

@author: mmdba
"""

import torch
import torch.nn as nn

class conv_patch_embed (nn.Module):
    def __init__(self, dim=768, in_channels=3, patch_size=16, image_size=224):
        super().__init__()
        
        self.dim = dim
        self.patch_size = patch_size
        self.n_patches = (image_size//patch_size)**2
        self.conv_layer = nn.Conv2d(in_channels, 
                                    dim, 
                                    kernel_size=patch_size,
                                    stride=patch_size
                                    )
    def forward(self,x):
        x = self.conv_layer(x) #(n_samples, dim, n_patches**0.5, n_patches**0.5)
        x = x.flatten(2)       #(n_samples, dim, n_patches)
        x = x.transpose(1,2)     #(n_samples, n_patches, dim)
        return x
        

class attention (nn.Module):
    def __init__(self, dim=768, n_heads=12, qkv_bias=True, att_drop_p=0., proj_drop_p=0.):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim//n_heads
        self.scale = (dim//n_heads)**0.5
        self.qkv_layer = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.proj = nn.Linear(dim,dim)
        self.drop_att = nn.Dropout(att_drop_p)
        self.drop_proj = nn.Dropout(proj_drop_p)
        
    def forward(self, x):
        n_samples, n_patches, dim = x.shape
        #arrert dim==self.dim 'the dimension of embedding doesnt match the network'
        x = self.qkv_layer(x) # (n_samples, n_patches, dim*3)
        x = x.reshape (n_samples, n_patches, 3, self.n_heads, self.head_dim)
        x = x.permute(2,0,3,1,4) # (3, n_samples, n_heads, n_patches, head_dim)
        q, k, v = x[0], x[1], x[2]   #(s_sample, n_heads, n_patches, head_dim)
        k_T = k.transpose(2,3)      #(s_sample, n_heads, head_dim, n_patches)
        att_map = ((q@k_T)/self.scale).softmax(dim=-1) #(s_sample, n_heads, n_patches, n_patches)
        attention_value = att_map@v  #(s_sample, n_heads, n_patches, head_dim)
        x = attention_value.transpose(1,2) #(s_sample, n_patches, n_heads, head_dim)
        x = x.flatten (2) #(s_sample, n_patches, dim)
        x = self.drop_att(x)
        x = self.proj(x)
        x = self.drop_proj(x)
        return x #(s_sample, n_patches, dim)

inp=torch.ones(32,196,768)
atten = attention()
out=atten(inp)
print(out.shape)
        
         
           
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        