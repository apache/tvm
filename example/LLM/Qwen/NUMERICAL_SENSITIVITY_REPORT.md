# Qwen 數值敏感性分析報告

## 問題摘要

**Qwen2.5-0.5B** 使用 Posit32 時與 FP32 有顯著差距（Token Match Rate: 25%）  
**Llama-3.2-1B** 使用 Posit32 時與 FP32 完美匹配（Token Match Rate: 100%）

## 根本原因

Qwen 對於數值精度**更加敏感**，主要原因是以下三個架構特性的組合：

### 1. 🔴 RMS Norm Epsilon（最關鍵）
| 模型 | Epsilon 值 | 差異 |
|------|-----------|------|
| **Qwen** | **1e-6** | 基準 |
| **Llama** | **1e-5** | **10倍大** |

**影響：**
- Epsilon 越小，對精度要求越高
- 當 RMS 值很小時，epsilon 主導分母，使得計算對誤差極其敏感
- 在極小值情況下可造成 **298% 的相對誤差**

### 2. 🟡 層數差異（高影響）
| 模型 | 層數 | 差異 |
|------|-----|------|
| **Qwen** | **24** | 基準 |
| **Llama** | **16** | **1.5倍少** |

**影響：**
- 每層累積小的數值誤差
- 累積誤差 ≈ 單層誤差 × √層數
- 24 層比 16 層多累積約 **23% 的誤差**

### 3. 🟠 GQA Ratio（中等影響）
| 模型 | GQA Ratio | 差異 |
|------|-----------|------|
| **Qwen** | **7.0** | 基準 |
| **Llama** | **4.0** | **1.75倍小** |

**影響：**
- 每個 KV pair 被重複使用 7 次 vs 4 次
- KV cache 中的誤差影響更多 query heads

## 誤差累積機制

```
Prefill 階段：
  FP32 → Posit32 轉換產生微小誤差
  ↓
  經過 RMS Norm (epsilon=1e-6) 放大誤差
  ↓
  誤差通過 24 層傳播
  ↓
  Prefill Logits MAE = 0.024（已可察覺）

Decode 階段 Step 1-3：
  使用帶誤差的 KV cache
  ↓
  誤差繼續累積但仍在閾值內
  ↓
  argmax 結果仍相同

Decode 階段 Step 4+：
  累積誤差超過閾值
  ↓
  Top logit 改變 → 不同的 token
  ↓
  完全不同的生成路徑
  ↓
  指數級發散
```

## 數值演示

當遇到很小的激活值時（RMS ≈ 2e-6）：

**Qwen (epsilon=1e-6):**
```
分母 = 2e-6 + 1e-6 = 3e-6
epsilon 佔分母的 33%
```

**Llama (epsilon=1e-5):**
```
分母 = 2e-6 + 1e-5 = 1.2e-5
epsilon 佔分母的 83%（更穩定）
```

**結果差異：298%！**

## 解決方案

### 方案 1：修改 Epsilon（最簡單）✅ 推薦
- 將 Qwen 的 epsilon 從 1e-6 改為 1e-5
- 需要重新編譯模型
- 對 FP32 準確度影響最小
- **預期可顯著改善 Token Match Rate**

### 方案 2：混合精度（最佳準確度）
- 權重使用 Posit32
- 僅 normalization layers 使用 FP32
- 防止關鍵操作中的誤差累積

### 方案 3：使用 Posit64（如果可行）
- 更高精度（~60 effective bits）
- 記憶體和計算成本更高

### 方案 4：選擇性 FP32（折衷方案）
- 前幾層使用 FP32
- 後續層使用 Posit32
- 防止早期誤差累積

## 驗證步驟

1. ✅ 已完成架構分析
2. ✅ 已確認 epsilon 差異
3. ⏳ 建議：修改 Qwen config，設定 epsilon=1e-5
4. ⏳ 重新編譯並測試
5. ⏳ 比較 Token Match Rate

## 結論

**Qwen 對數值更敏感**的原因：
- ✗ 10倍更小的 normalization epsilon (1e-6 vs 1e-5)
- ✗ 50% 更多的層數 (24 vs 16)
- ✗ 更高的 GQA ratio (7 vs 4)

這是一個**設計敏感性問題**，而非 Posit32 算術的根本限制。
通過調整 epsilon，Qwen 應該能很好地使用 Posit32。

---

**分析完成日期：** 2025-12-03  
**分析工具：** 
- `analyze_numerical_sensitivity.py`
- `check_normalization_precision.py`
- `root_cause_summary.py`
