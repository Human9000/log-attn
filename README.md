# nd-log-flash-attn
本文提出了一种名为对数注意力（Logarithmic Attention, LogA）的高效注意力机制，其设计灵感源自矩阵初等变换，并基于多维对数分解，能够处理任意维度的输入数据。LogA 首先对输入数据进行展平，通过创新的多维对数分解策略，将展平数据映射至多维张量表示。随后，在各个维度上施加类似初等变换的注意力操作，高效聚合多维信息，构建全局注意力机制，并实现对数级别的计算复杂度。此外，LogA 引入 Flash-Attention 技术，有效优化了 $\mathop{SoftMax}(\mathbf{qk}^\top)\mathbf{v}$ 操作的 I/O 开销，并显著降低显存占用，从而进一步提升了计算效率和资源利用率。


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
