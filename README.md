# nd-log-flash-attn
本文提出了一种名为对数注意力（Logarithmic Attention，LogA）的高效注意力机制，其灵感来源于矩阵初等变换，并基于多维对数分解来处理任意维度输入数据。类似于通过行变换和列变换实现任意矩阵变换，LogA 首先将输入数据展平，然后利用一种创新的多维对数分解策略，将展平后的数据映射至多维张量表示，并在各个维度上应用类似初等变换的注意力操作，从而有效捕捉跨维度特征交互。LogA 通过利用这种多维张量表示，高效地聚合多维局部注意力，构建全局注意力，并实现了对数级的计算复杂度。为了进一步提升效率并降低显存占用，LogA 引入了 Flash-Attention 来优化 Softmax($\mathbf{qk}^\top$)$\mathbf{v}$ 操作的 I/O 成本，使其更适用于大规模应用场景。


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
