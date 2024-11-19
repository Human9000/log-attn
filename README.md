# nd-log-flash-attn
本文提出了一种名为N维对数注意力（N-dimensional Logarithmic Flash Attention，MLFA）的创新性注意力机制，能够高效处理任意维度的输入数据。MLFA 的核心在于其通用的多维对数分解策略，该策略将任意维度输入数据映射至一个相应的多维张量。这种多维表示允许我们并行地计算各个维度上的紧凑初等变换矩阵，有效捕捉不同维度特征之间的细粒度交互。这些多维局部注意力随后被聚合，以获得全局注意力，从而实现对数级的复杂度降低。此外，MLFA 无缝集成了 Flash-Attention 的高性能分块计算方法，进一步降低了 Softmax(qk)v 操作中的 I/O 开销，显著提升了计算效率。


# 环境参数
```bash
flash-attn==2.7.0.post2
ptflops
torch
```

# 安装
```bash
git clone https://github.com/Human9000/nd-log-flash-attn.git
cd nd-log-flash-attn
pip install .
```

# 测试
```bash
python demo/test_flops_params.py
```
