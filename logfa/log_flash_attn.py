
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

from flash_attn import flash_attn_func, flash_attn_qkvpacked_func

# 注册到ptflop的参数量计算中
from ptflops.pytorch_engine import MODULES_MAPPING 
class FlashAttnFlopsHook(nn.Module):
    def __init__(self, ):
        super().__init__()
        def pass_through(*args, **kwargs): return
        MODULES_MAPPING[FlashAttnFlopsHook] = pass_through

    def forward(self, qkv):
        b, l, _, h, d = qkv.shape
        self.__flops__ += 2 * b * l * l * h * d  # 统计qkv的计算量
        return qkv


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    default_act = nn.SiLU()  # default activation

    def __init__(self, dim, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        _Conv = [nn.Conv1d, nn.Conv2d, nn.Conv3d][dim-1]
        _BN = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d][dim-1]

        self.conv = _Conv(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = _BN(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))
 

class UpsampleLike(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mode = ['linear', 'bilinear', 'trilinear'][dim-1]

    def forward(self, x, like):
        if x.shape[2:] == like.shape[2:]:
            return x
        return F.interpolate(x, size=like.shape[2:], mode=self.mode)


class LogFlashAttentionNd(nn.Module):
    def __init__(self, dim, in_channels, num_heads=8, bases=[16, 5], down_factor=1, regist_flops=False):
        super().__init__()

        AvgPool = [nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d][dim-1]
        self.dim = dim
        head_dim = in_channels // num_heads
        self.static_params = [num_heads, head_dim,]
        h = (head_dim * num_heads) * 2  # 2 for query and key
        self.bases = bases
        self.down_factor = down_factor
        self.qks = nn.ModuleList(Conv(dim, in_channels, h, 1, act=False) for _ in self.bases)
        self.vs = nn.ModuleList(Conv(dim, in_channels, in_channels, 1, act=False) for _ in self.bases)
        self.pes = nn.ModuleList(Conv(dim, in_channels, in_channels, 3, 1, g=in_channels, act=False) for _ in self.bases)

        self.avgpool_qk = AvgPool(down_factor, down_factor) if down_factor > 1 else nn.Identity()
        self.avgpool_v = AvgPool(down_factor, down_factor) if down_factor > 1 else nn.Identity()
        self.upsample_like = UpsampleLike(dim=dim) 
        self.proj = Conv(dim, in_channels, in_channels, 1, act=False)
        self.cuda_batch_size = 2**16 - 1  # 4090 GPU allow max batch size 
        self.flash_attn_flops_hook = FlashAttnFlopsHook() if regist_flops else nn.Identity()

    def _log_1d_attn_func(self, qk, v, base):
        h, d = self.static_params
        b, c, w = v.shape 
        n = int(math.log(w, base) - 1e-9) + 1
        qk = F.pad(qk, (0, base**n - w), value=0.0)  # pad到base的整数次幂
        v = F.pad(v, (0, base**n - w), value=0.0)  # pad到base的整数次幂
        qk = qk.transpose(1, 2).view(b, *[base,]*n, 2, h, d)
        v = v.transpose(1, 2).view(b, *[base,]*n, 1, h, d,)   # （batch, *seqlen,  nheads, headdim )
        cbs = self.cuda_batch_size
        for i in range(n):
            qkv = torch.cat([qk, v], dim=-3).transpose(i+1, -4).flatten(0, -5).half()  # (batch * batch_len, seqlen, nheads, headdim,)
            ibs = qkv.shape[0]
            v2_list = []
            for j in range((ibs + cbs - 1)//cbs):  # 不超出GPU的最大线程数
                v2_list.append(flash_attn_qkvpacked_func(qkv[j*cbs:(j+1)*cbs], 0.0))
                self.flash_attn_flops_hook(qkv)  # 统计flash_attn的计算量
            v = torch.cat(v2_list, dim=0) .unflatten(0, [b, ]+[base,]*(n-1))
            v = v.transpose(i+1, -4).unsqueeze(-3)

        # 删除pad的部分，恢复到原来的形状
        v = v.reshape(b, -1, c)[..., :w, :].transpose(1, 2)
        return v

    def _log_attn_func(self, x, index):
        qk = self.qks[index]
        v = self.vs[index]
        pe = self.pes[index]
        base = self.bases[index]

        qk = qk(x)  # 提取query和key
        v = v(x)    # 提取value
        pe = pe(v)  # 提取位置编码
        # print(qk.shape, v.shape, pe.shape)
        # downsample,平滑局部信息，减少计算量
        qk = self.avgpool_qk(qk)
        v = self.avgpool_v(v)

        y = self._log_1d_attn_func(qk.flatten(2),  # 多维度数据拉直成1d向量
                                   v.flatten(2),  # 多维度数据拉直成1d向量
                                   base).unflatten(2, qk.shape[2:])  # 还原拉直的多维空间维度

        # upsample,平滑局部信息，减少计算量
        y = self.upsample_like(y, pe)

        return y + pe  # 添加位置编码

    def forward(self, x):
        y = x
        # 多个base下，多次计算累加
        for index in range(len(self.bases)):
            y = y + self._log_attn_func(x, index)
        # 最后输出做通道投影
        return self.proj(y)


class LogFA1d(LogFlashAttentionNd):
    def __init__(self, nheads, headdim, bases, down_factor=1, regist_flops=False):
        super().__init__(1, nheads, headdim, bases, down_factor=down_factor, regist_flops=regist_flops)


class LogFA2d(LogFlashAttentionNd):
    def __init__(self, nheads, headdim, bases, down_factor=1, regist_flops=False):
        super().__init__(2, nheads, headdim, bases, down_factor=down_factor, regist_flops=regist_flops)


class LogFA3d(LogFlashAttentionNd):
    def __init__(self, nheads, headdim, bases, down_factor=1, regist_flops=False):
        super().__init__(3, nheads, headdim, bases, down_factor=down_factor, regist_flops=regist_flops)

