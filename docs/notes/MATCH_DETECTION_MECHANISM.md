# run_two_pass.sh 中的 Match 检测机制详解

## 📋 概述

`run_two_pass.sh` 脚本通过 **两阶段搜索 + 验证** 的方式来检测公式是否与 Ground Truth (GT) 匹配：

- **Pass 1**：无预处理、无精化的正常搜索
- **验证阶段**：调用 `verify.py` 检查 Pass 1 的结果
- **Pass 2**：对未匹配的方程，启用预处理和精化后重新搜索

---

## 🔍 Match 检测的三层结构

### 第 1 层：日志文件标记（Marker-based）

#### 脚本中的检查函数

```bash
# Line 51-53
is_matched() {
    local logfile="$1"
    grep -q 'Conclusion: Match found' "$logfile" 2>/dev/null
}

# Line 45-47
is_completed() {
    local logfile="$1"
    grep -q 'Search completed' "$logfile" 2>/dev/null
}
```

**检测原理**：
- `is_matched()` 检查日志末尾是否包含字符串 **`Conclusion: Match found`**
- `is_completed()` 检查日志是否包含 **`Search completed`**

### 第 2 层：验证函数的调用流程

```bash
# Line 68-87: verify_equation() 函数
verify_equation() {
    local eq_tag="$1" logfile="$2"
    
    # 第一步：检查是否已有 Match 标记
    if is_matched "$logfile"; then
        echo "  [MATCHED] $eq_tag"
        return 0  # 返回码 0 = 已匹配
    fi
    
    # 第二步：调用 verify.py 进行 LLM 验证
    echo -n "  [VERIFYING] $eq_tag ... "
    local verify_output
    verify_output=$(python verify.py "$logfile" --llm-config "$VERIFY_CONFIG" 2>&1) || true

    # 第三步：检查 LLM 验证结果
    if echo "$verify_output" | grep -q '\[MATCH FOUND\]'; then
        echo "Match found"
        # 将验证结果附加到日志文件
        printf '\n%s\n' "$verify_output" >> "$logfile"
        return 0  # 返回码 0 = 匹配
        
    elif echo "$verify_output" | grep -q '\[verify error\]'; then
        local reason
        reason=$(echo "$verify_output" | grep '\[verify error\]' | head -1)
        echo "Skipped ($reason)"
        return 2  # 返回码 2 = 跳过（错误）
        
    else
        echo "No match"
        printf '\n%s\n' "$verify_output" >> "$logfile"
        return 1  # 返回码 1 = 不匹配
    fi
}
```

**返回码含义**：
- `0` = 匹配（`MATCHED`）
- `1` = 不匹配（`NO MATCH`）
- `2` = 跳过（验证错误）

### 第 3 层：具体实例（以 BPG0 为例）

根据你提供的日志文件 `BPG0_20260328_031923.txt`：

#### ✅ 包含的关键标记

```
Line 1304: Search completed, elapsed 1925.9s
Line 1636: [MATCH FOUND]
Line 1644: Conclusion: Match found
```

#### 输出示例

从日志末尾的验证结果：

```
============================================================
GT Equivalence Verification (BPG0)
============================================================
GT: 0.9539969835279999*(1 - P/96.90688297671034)*P + 0.9539969835279999*P**0.333333333333333
Candidate formulas: 51

[MATCH FOUND]

This is a perfect match!

Let me also check some others that might match:

**Candidate 50:** `P**2*c0 + P*c1 + P**c2*c3` = `P²*c0 + P*c1 + P**c2*c3`
With appropriate choices of c0, c1, c2, c3, this form can represent the target polynomial.
This is a perfect match!

Multiple candidates match, but the clearest and most direct match is:

**Candidate 50:** `P**2*c0 + P*c1 + P**c2*c3`

** Other matching candidates include 41, 42, 46 (and others with redundant terms that reduce to the same form).

Conclusion: Match found
============================================================
```

---

## 🔄 完整的检测流程

### Pass 1 + 验证 的执行流程

```
┌─────────────────────────────────────┐
│ 1. 执行 Pass 1 搜索                  │
│    python main.py --dataset ...     │
│    → 生成日志: BPG0_20260328_031923.txt│
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 2. 收集已完成的方程                   │
│    collect_completed_equations()    │
│    检查：grep 'Search completed'    │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 3. 对每个方程执行 verify_equation()  │
└──────────────┬──────────────────────┘
               │
        ┌──────┴─────┐
        │             │
        ▼             ▼
   ┌─────────┐   ┌─────────────┐
   │ 已有    │   │ 需要验证    │
   │ Match   │   │             │
   │ 标记    │   └────┬────────┘
   │ ?       │        │
   │ YES     │        ▼
   └────┬────┘   ┌─────────────────────┐
        │        │ 调用 verify.py      │
        │        │ 提交 LLM 判断等价性  │
        │        │ 检查：[MATCH FOUND] │
        │        └────┬────────────────┘
        │             │
        │      ┌──────┴──────┐
        │      │             │
        │      ▼             ▼
        │   ┌─────┐      ┌──────┐
        │   │Match│      │ No   │
        │   │ (0) │      │Match │
        │   └─────┘      │ (1)  │
        │      │         └──────┘
        └──────┴──────────┬──────┘
                         │
                         ▼
        ┌─────────────────────────┐
        │ 统计结果                │
        │ MATCHED: N              │
        │ UNMATCHED: M            │
        └─────────┬───────────────┘
                  │
         ┌────────┴────────┐
         │                 │
    ┌────▼─────┐      ┌───▼─────┐
    │ M == 0   │      │ M > 0   │
    │ 无需Pass2│      │需Pass 2 │
    │ 退出     │      │继续     │
    └──────────┘      └─────────┘
```

---

## 🧪 实例讲解：BPG0 的检测过程

### 第 1 步：搜索完成标记

日志中显示：
```
Line 1304: Search completed, elapsed 1925.9s
```

检查代码（run_two_pass.sh）：
```bash
is_completed "$logfile"  # 检查是否有 'Search completed'
→ grep -q 'Search completed' "$logfile"
→ 返回 0 (TRUE)
```

✅ **判断**：搜索已完成，可以进行验证

### 第 2 步：初次 Match 检查

检查代码（verify_equation()）：
```bash
if is_matched "$logfile"; then
    echo "  [MATCHED] $eq_tag"
    return 0
fi
```

对应的 grep：
```bash
grep -q 'Conclusion: Match found' "$logfile"
→ 搜索到第 1644 行的标记
→ 返回 0 (TRUE)
```

✅ **判断**：日志中已经有了验证结果，直接输出 `[MATCHED] BPG0`

### 第 3 步：Pass 1 验证统计

```bash
for eq_tag in $(echo "${!EQ_LOGS[@]}" | tr ' ' '\n' | sort); do
    logfile="${EQ_LOGS[$eq_tag]}"
    verify_equation "$eq_tag" "$logfile"
    rc=$?
    if [ $rc -eq 0 ]; then
        ((MATCHED++)) || true
    elif [ $rc -eq 2 ]; then
        ((SKIPPED++)) || true
    else
        FAILED_EQUATIONS+=("$eq_tag")
    fi
done
```

**统计结果**：
- BPG0: `rc=0` → `MATCHED++`

### 第 4 步：决定是否执行 Pass 2

```bash
if [ ${#FAILED_EQUATIONS[@]} -eq 0 ]; then
    echo "All equations matched GT, no need for Pass 2."
    exit 0
fi
```

**判断**：
- `FAILED_EQUATIONS` 数组为空
- 所有方程都已匹配
- **退出**，无需 Pass 2

---

## 🔧 verify.py 的内部机制

### 验证流程图

```
┌──────────────────────────────┐
│ verify.py <log_path>         │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ 1. 提取方程名称               │
│    extract_equation_name()   │
│    从日志中找 "equation=BPG0" │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ 2. 加载 Ground Truth          │
│    load_gt(equation_name)    │
│    从 gt_expressions.csv      │
│    获得 GT 符号表达式         │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ 3. 提取候选公式               │
│    extract_formulas()        │
│    从日志中找 <<<FORMULA>>>...│
│                              │
│    BPG0 结果：51 个公式       │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ 4. 调用 LLM 判断等价性         │
│    build_prompt()            │
│    构建提示：                 │
│    - GT: 目标表达式           │
│    - 51 个候选公式            │
│    - 等价性规则               │
│    call_llm()                │
│    调用 Verifier LLM          │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ 5. 解析 LLM 响应              │
│    检查 [MATCH FOUND] 标记    │
│    → Conclusion: Match found  │
│    检查 [NO MATCH] 标记       │
│    → Conclusion: No match     │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ 6. 返回结果                   │
│    返回验证文本               │
│    run_two_pass.sh 将其       │
│    附加到原日志末尾           │
└──────────────────────────────┘
```

### verify.py 的关键代码

#### 提取公式

```python
# Line 60-67
_FORMULA_TAG = re.compile(r"<<<FORMULA>>>(.+?)<<<END_FORMULA>>>")

def extract_formulas(log_path: str) -> list[str]:
    text = Path(log_path).read_text(encoding="utf-8")
    tagged = _FORMULA_TAG.findall(text)
    if tagged:
        return [f.strip() for f in tagged if f.strip()]
```

**作用**：从日志中找到形如 `<<<FORMULA>>>P**2*c0 + P*c1<<<END_FORMULA>>>` 的公式

#### 构建 LLM 提示

```python
# Line 107-129
def build_prompt(gt_symbolic: str, gt_features: str, formulas: list[str]) -> str:
    return f"""...
## Equivalence Rules
- The constants c0, c1, c2, ... can take any real value (including 0).
- Two formulas are equivalent if there exists an assignment of constants 
  such that the two formulas are equal for all values of feature variables.
...
"""
```

**作用**：告诉 LLM 如何判断等价性

#### 检查 LLM 响应

```python
# Line 209-214
if "[MATCH FOUND]" in answer:
    conclusion = "Match found"
elif "[NO MATCH]" in answer:
    conclusion = "No match"
else:
    conclusion = "Unable to parse (LLM did not return explicit tag)"
```

**作用**：解析 LLM 的第一行标记

---

## 📊 检测状态图

| 日志中的标记 | verify_equation 返回码 | 含义 |
|----------|--------|------|
| `Conclusion: Match found` 已存在 | `0` | ✅ 已匹配 |
| 需要调用 verify.py，LLM 返回 `[MATCH FOUND]` 并写入 `Conclusion: Match found` | `0` | ✅ 新匹配 |
| 需要调用 verify.py，LLM 返回 `[NO MATCH]` 并写入 `Conclusion: No match` | `1` | ❌ 不匹配 |
| verify.py 出错，返回 `[verify error]` | `2` | ⚠️ 跳过 |

---

## 💡 核心检测机制总结

### 1. **两阶段检测**

```bash
┌──────────────────────────────────┐
│ 快速检测：grep 日志标记           │
│ grep -q 'Conclusion: Match found'│
│ ✅ O(1) 复杂度，毫秒级           │
└──────────────────────────────────┘
           ↓ (如果未找到)
┌──────────────────────────────────┐
│ 深度检测：调用 LLM 验证           │
│ verify.py → LLM 等价性判断        │
│ ⚙️ O(n) 复杂度，秒级             │
└──────────────────────────────────┘
```

### 2. **三层验证标记**

```
搜索完成 → "Search completed"
  ↓
验证结果 → "[MATCH FOUND]" 或 "[NO MATCH]"
  ↓
最终结论 → "Conclusion: Match found/No match"
```

### 3. **返回码体系**

```bash
verify_equation() 的返回码规范：
- 0 = MATCHED   → 计数器 MATCHED++
- 1 = UNMATCHED → 加入 FAILED_EQUATIONS 数组
- 2 = SKIPPED   → 计数器 SKIPPED++
```

---

## 🎯 BPG0 案例的检测流程图

```
搜索日志: BPG0_20260328_031923.txt (1644 行)
    │
    ├─ [搜索完成标记] Line 1304: "Search completed"
    │    └─ is_completed() → TRUE ✅
    │
    ├─ [验证部分] Line 1612 onwards
    │    ├─ "[MATCH FOUND]" at Line 1636
    │    └─ "Conclusion: Match found" at Line 1644
    │         └─ is_matched() → TRUE ✅
    │
    └─ [最终判断]
         verify_equation("BPG0", log) → 0
         MATCHED++ → MATCHED=1
         FAILED_EQUATIONS 保持为空
         → 无需 Pass 2，直接退出
```

---

## 📝 总结

`run_two_pass.sh` 的 Match 检测分为三个层级：

1. **快速检查**（第一次调用）：`grep` 查找 `Conclusion: Match found`
2. **深层验证**（第二次调用）：调用 `verify.py` 执行 LLM 等价性判断
3. **统计决策**：基于返回码统计匹配/不匹配的方程数，决定是否执行 Pass 2

对于 BPG0 的例子，日志中已包含完整的验证结果，所以脚本直接识别为"已匹配"，无需重新调用 LLM。

