import numpy as np

# ====================== Generate random test data ======================
# Define 10x10 weight matrix with values from 1-100
W = np.random.rand(1024, 1024).astype(np.float32)

X = np.random.rand(1024, 1024).astype(np.float32)  # Generate 10x10 random matrix with values between 0-1

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

# ====================== GPTQ量化（基于Hessian矩阵调整） ======================
def gptq_quantize(W, X, bits=3, block_size=2, damp=0.01):
    """GPTQ量化函数"""
    rows, cols = W.shape
    W_quant = W.copy()
    
    X_proc = np.expand_dims(X, axis=0).reshape(-1, X.shape[-1])
    
    # 计算Hessian矩阵并添加阻尼项
    H = X_proc @ X_proc.T
    H += damp * np.eye(rows)  # 确保矩阵可逆    
    H_inv = np.linalg.inv(H)  # Hessian逆矩阵
    
    # 逐列量化（按block_size分块）
    for col_start in range(0, cols, block_size):
        col_end = min(col_start + block_size, cols)
        H_inv_sub = H_inv[col_start:col_end, col_start:col_end]
        
        # 处理当前块内的每一列
        for c in range(col_start, col_end):
            # 量化当前列的权重
            # quant_val = np.round(W_quant[:, c])
            quant_val = rtn_quantize(W_quant[:, c].reshape(-1, 1), bits=bits, group_size=1).flatten()
            error_start2c = (W_quant[:, c].flatten() - quant_val) / H_inv[c, c]  # 量化误差
            
            if block_size == 1:
                W_quant[:, c] = np.squeeze(quant_val)
                continue #> no need to update
            
            # 调整剩余列权重以补偿误差
            if c <= col_end:
                # 提取Hessian逆矩阵的对应子块（从c行c列开始，取剩余部分）
                
                # 计算补偿量：delta * H_inv的对应行
                # 维度修正：delta (rows,) -> (rows, 1), H_sub_inv (remaining_cols, remaining_cols)

                update = error_start2c.reshape(-1, 1) @ np.expand_dims(H_inv[c, c:col_end], 0)
                # 更新后续列权重
                W_quant[:, c:col_end] -= update
            
            # 应用量化后的值
            W_quant[:, c] = quant_val
    
    return W_quant

# W_gptq = gptq_quantize(W, X, bits=3, block_size=2)

# ====================== 计算输出MSE ======================
def compute_output_mse(W, W_quant, X):
    """计算量化后的输出MSE"""
    Y_original = W @ X.T  # 原始输出
    Y_quant = W_quant @ X.T  # 量化后输出
    return np.mean((Y_original - Y_quant) ** 2)

W_rtn = rtn_quantize(W, bits=3, group_size=1)
mse_rtn = compute_output_mse(W, W_rtn, X)

W_gptq = gptq_quantize(W, X, bits=3, block_size=16)
mse_gptq = compute_output_mse(W, W_gptq, X)

# ====================== 结果对比 ======================
print("原始权重矩阵:\n", W)
print("\n普通量化（RTN）结果:\n", W_rtn)
print("\n普通量化输出MSE:", mse_rtn)
print("\nGPTQ量化结果:\n", W_gptq)
print("\nGPTQ量化输出MSE:", mse_gptq)
