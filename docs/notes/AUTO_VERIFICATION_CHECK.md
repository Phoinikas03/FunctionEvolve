# Main.py 自动验证功能核实报告

## ✅ 代码层面的核实

### 1. main.py 中确实有自动验证代码

**位置**：[main.py](main.py#L600-L611)

```python
# ---- 7. GT equivalence verification ----
if llm_config:
    try:
        from verify import verify_log
        verify_text = verify_log(log_path, llm_config, verbose=verbose)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(verify_text + "\n")
        if verbose:
            print(f"[verify] Verification results appended to log: {log_path}")
    except Exception as e:
        if verbose:
            print(f"[verify error] GT verification failed: {e}")
```

**执行时机**：在主搜索完成 **之后** 立即执行

```python
Line 567:  searcher.run()                    # ← 搜索完成
Line 595:  searcher.close_log()              # ← 关闭搜索阶段的日志
           ...输出最终结果...
Line 600:  # ---- 7. GT equivalence verification ----  # ← 自动验证开始
Line 603:  verify_text = verify_log(...)     # ← 调用verify.py
Line 606:  f.write(verify_text + "\n")       # ← 附加到日志
```

---

## 🔍 BPG0 日志中的验证痕迹分析

### 日志结构

```
Line 1304:  "Search completed, elapsed 1925.9s"
            ↓
Line 1320:  "Final search results"
            ← 搜索输出完成
            ↓
Line 1584:  "GT Equivalence Verification (BPG0)"
            ← 验证结果出现
            ↓
Line 1644:  "Conclusion: Match found"
            ← 最后一行
```

### ⚠️ 关键观察

**验证结果确实被附加到了日志**，但：

- ✅ 存在验证文本（Line 1584-1644）
- ✅ 存在 "Conclusion: Match found" 标记
- ❌ **没有** `[verify]` 前缀的日志
- ❌ **没有** `[verify error]` 错误信息

---

## 🔧 可能的原因

### 情况 A：使用了 main.py 的自动验证

如果是这样，应该看到：
```
[verify] Equation name: BPG0
[verify] GT symbolic expression: ...
[verify] Feature variables: ...
[verify] Extracted XX unique candidate formulas
[verify] Prompt length: XXXXX characters
[verify] Calling LLM for equivalence check...
[verify] Verification results appended to log: ...
```

**但日志中没有这些**。

### 情况 B：使用了 run_two_pass.sh 的 verify.py

如果是这样（因为这是目前的默认流程）：
```bash
$ python verify.py logs/llm-srbench/full/opus-4-6/BPG0.txt --llm-config gemini-3-pro-for-verifier.yaml
```

这样的话：
- ✅ 验证结果会被附加
- ✅ 但不会有 `[verify]` 输出（除非 verbose 模式）
- ✅ 符合当前日志的样子

---

## 📋 确认清单

要验证 main.py 自动验证是否工作，需要：

### 1. 检查 main.py 代码版本
是否包含第 600-611 行的验证代码？

**结果**： ✅ **有**

```python
# ---- 7. GT equivalence verification ----
if llm_config:
    try:
        from verify import verify_log
        verify_text = verify_log(log_path, llm_config, verbose=verbose)
        ...
```

### 2. 检查日志生成方式
- 如果是 `python main.py ...` 直接运行
  → 应该有自动验证
- 如果是 `run_two_pass.sh` 运行
  → 验证由脚本的 verify_equation() 负责

**对于 BPG0 日志**：
- Log file: `logs/llm-srbench/full/opus-4-6/BPG0.txt`
- Model: Opus-4-6
- 这个日志可能是通过 `run_two_pass.sh` 生成的

### 3. 运行日志中的线索
- ✅ 验证文本存在
- ❌ 但没有看到启动时的验证日志输出

---

## 💡 结论

**主要发现**：

| 方面 | 状态 | 证据 |
|------|------|------|
| main.py 中有验证代码 | ✅ **确认** | [main.py](main.py#L600-L611) 第 600-611 行 |
| BPG0 日志包含验证结果 | ✅ **确认** | Line 1584-1644 "GT Equivalence Verification" |
| 验证是 main.py 执行的 | ❓ **不确定** | 日志中没有 `[verify]` 输出 |

---

## 🎯 建议验证步骤

### 方案 1：查看最近的搜索日志

```bash
# 运行一次新的搜索
python main.py --dataset bio_pop_growth --equation BPG0 --llm-config gpt-5-2.yaml -v

# 检查日志中是否包含：
# [verify] Equation name: BPG0
# [verify] GT symbolic expression: ...
# [verify] Verification results appended to log: ...
```

### 方案 2：检查 main.py 的执行流程

确认搜索完成后，main.py 是否：
1. 调用了 `verify_log()`
2. 将结果附加到日志
3. 输出 `[verify]` 前缀的日志

### 方案 3：对比日志特征

```
main.py 自动验证的日志特征：
- 验证发生在 "Search completed" 之后
- 日志末尾包含 "[verify]" 前缀的日志
- 最后两行：
  [verify] Verification results appended to log: ...
  Conclusion: Match found

run_two_pass.sh 验证的日志特征：
- 验证发生在脚本中独立调用 verify.py
- 日志末尾可能没有 "[verify]" 前缀
- 但包含 "GT Equivalence Verification (BPG0)" 的完整验证文本
```

---

## 📝 总结

✅ **代码层面**：main.py 已经实现了自动验证功能（第 600-611 行）

❓ **实际执行**：BPG0 日志中无法确认是 main.py 还是 run_two_pass.sh 执行的验证

**下一步**：建议运行新的搜索任务，观察输出日志中是否包含 `[verify]` 前缀的消息，以确认 main.py 的自动验证是否被执行。

