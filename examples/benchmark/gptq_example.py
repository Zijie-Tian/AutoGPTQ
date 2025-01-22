import torch
import torch.nn as nn
import numpy as np

# Create GPTQ instance
from auto_gptq.quantization import GPTQ

# ====================== Generate random test data ======================
# Define 10x10 weight matrix with values from 1-100
W_raw = np.random.rand(8, 8).astype(np.float32)

X_raw = np.random.rand(10, 8).astype(np.float32)  # Generate 10x10 random matrix with values between 0-1

# ====================== Group-wise 量化 ======================
def rtn_quantize(weight, bits=3, group_size=32):
    """Group-wise 四舍五入量化"""
    #> is 2 ** (bits - 1) - ( - 2 ** (bits - 1))
    quant_max = 2 ** (bits - 1) - 1
    quant_min = -2 ** (bits - 1)
    scale = (quant_max - quant_min)  # 3-bit范围: -4到3（对称量化）
    rows, cols = weight.shape
    
    # 初始化量化结果
    quantized = np.zeros_like(weight)
    
    # 按group_size分组量化
    for g in range(0, cols, group_size):
        group_end = min(g + group_size, cols)
        group = weight[:, g:group_end]

        # 计算每个group的scale
        group_max = np.max(np.abs(group))
        group_min = np.min(group)
        group_scale = (group_max - group_min) / scale
        
        # 量化当前group
        quantized_group = np.clip(np.round(group * (1.0 / group_scale)), quant_min, quant_max)
        # 反量化并存储
        quantized[:, g:group_end] = quantized_group * group_scale

    return quantized

def gptq_quantize(W, X, bits=3, block_size=2):
    """GPTQ量化"""
    # Convert numpy arrays to torch tensors
    W_torch = torch.from_numpy(W)
    X_torch = torch.from_numpy(X)

    # Create linear layer and initialize with W
    linear = nn.Linear(W.shape[1], W.shape[0], bias=False)
    linear.weight.data = W_torch

    gptq = GPTQ(linear, bits=bits)

    gptq.quantizer.configure(bits=bits)

    gptq.add_batch(X_torch, None)

    # Run GPTQ quantization
    gptq.fasterquant(blocksize=block_size, percdamp=0.01)
    
    # Get quantized weights back as numpy array
    W_quant = linear.weight.data.cpu().numpy()
    
    return W_quant


# ====================== GPTQ量化（基于Hessian矩阵调整） ======================
def gptq_simple_quantize(W, X, bits=3, block_size=2, damp=0.01):
    """ 复现GPTQ量化函数"""
    from auto_gptq.quantization.quantizer import Quantizer
    
    quantizer = Quantizer()
    quantizer.configure(bits=bits)

    if not quantizer.ready():
        quantizer.find_params(torch.from_numpy(W), weight=True)

    rows, cols = W.shape
    W_quant = np.zeros_like(W)
    
    X_proc = np.expand_dims(X, axis=0).reshape(-1, X.shape[-1])
    
    # import time 
    # start = time.time()
    # # 计算Hessian矩阵并添加阻尼项
    H = X_proc.T @ X_proc
    # H += damp * np.eye(cols)  # 确保矩阵可逆    
    # H_inv = np.linalg.inv(H)  # Hessian逆矩阵
    # end = time.time()
    # print(f"Hessian矩阵计算时间: {end - start}")

    scale = []
    zero = []
    H = torch.from_numpy(H)
    diag = torch.arange(cols, device='cpu')
    H[diag, diag] += damp
    H = torch.linalg.cholesky(H)
    H = torch.cholesky_inverse(H)
    H = torch.linalg.cholesky(H, upper=True)
    H_inv = H.cpu().numpy()
    
    # 逐列量化（按block_size分块）
    for col_start in range(0, cols, block_size):
        col_end = min(col_start + block_size, cols)

        W_local = np.copy(W[:, col_start:col_end])
        Q_local = np.zeros_like(W_local)
        Err_local = np.zeros_like(W_local)
        H_inv_local = H_inv[col_start:col_end, col_start:col_end]
        
        # 处理当前块内的每一列
        # for c in range(col_start, col_end):
        for c in range(col_end - col_start):
            w = W_local[:, c]
            d = H_inv_local[c, c]
            
            # 量化当前列的权重
            # quant_val = np.round(W_quant[:, c])
            # quant_val = rtn_quantize(W[:, c].reshape(-1, 1), bits=bits, group_size=1).flatten()

            if (col_start + c) % block_size == 0:
                quantizer.find_params(torch.from_numpy(W[:, col_start:col_end]), weight=True)            
            
            quant_val = quantizer.quantize(np.expand_dims(w, axis=1)).flatten()
            
            # import pdb; pdb.set_trace()
            
            error_start2c = (w - quant_val.numpy()) / d  # 量化误差

            if block_size == 1:
                W_quant[:, c] = np.squeeze(quant_val)
                continue #> no need to update
            
            # 调整剩余列权重以补偿误:
            # 提取Hessian逆矩阵的对应子块（从c行c列开始，取剩余部分）
            Q_local[:, c] = quant_val

            update = error_start2c.reshape(-1, 1) @ np.expand_dims(H_inv_local[c, c:col_end], 0)
            Err_local[:, c] = error_start2c
            # 更新后续列权重
            W_local[:, c:col_end] -= update

        W_quant[:, col_start:col_end] = Q_local
        W[:, col_end:] -= Err_local @ H_inv[col_start:col_end, col_end:]
    
    return W_quant


# W_gptq = gptq_quantize(W, X, bits=3, block_size=2)

# ====================== 计算输出MSE ======================
def compute_output_mse(W, W_quant, X):
    """计算量化后的输出MSE"""
    Y_original = W @ X.T  # 原始输出
    Y_quant = W_quant @ X.T  # 量化后输出
    return np.mean((Y_original - Y_quant) ** 2)

import time

W = np.copy(W_raw)
X = np.copy(X_raw)

start = time.time()
W_rtn = rtn_quantize(W, bits=3, group_size=1)
end = time.time()
print(f"RTN量化时间: {end - start}")

mse_rtn = compute_output_mse(W, W_rtn, X)

W = np.copy(W_raw)
X = np.copy(X_raw)

start = time.time()
W_gptq = gptq_quantize(W, X, bits=3, block_size=2)
end = time.time()
print(f"GPTQ量化时间: {end - start}")

W = np.copy(W_raw)
X = np.copy(X_raw)
mse_gptq = compute_output_mse(W, W_gptq, X)
print("\nGPTQ量化结果:\n", W_gptq)
print("\nGPTQ量化输出MSE:", mse_gptq)

W = np.copy(W_raw)
X = np.copy(X_raw)

start = time.time()
W_gptq_simple = gptq_simple_quantize(W, X, bits=3, block_size=2)
end = time.time()
print(f"GPTQ简单量化时间: {end - start}")

W = np.copy(W_raw)
X = np.copy(X_raw)
mse_gptq_simple = compute_output_mse(W, W_gptq_simple, X)
print("\nGPTQ简单量化结果:\n", W_gptq_simple)
print("\nGPTQ简单量化输出MSE:", mse_gptq_simple)

# ====================== 结果对比 ======================
# print("原始权重矩阵:\n", W)
# print("\n普通量化（RTN）结果:\n", W_rtn)
# print("\n普通量化输出MSE:", mse_rtn)
