# verify.py 给 LLM Judge 的输入信息详解

## 📋 总体流程

```
BPG0 Search Log
    ↓
1. extract_equation_name()  → 从日志提取 "BPG0"
    ↓
2. load_gt(eq_name)         → 从 CSV 加载 GT 信息
    ↓
3. extract_formulas()       → 从日志提取 87 个候选公式
    ↓
4. build_prompt()           → 构建 LLM 确定 ← 你走的是这一步
    ↓
5. call_llm()               → 发送给 Verifier LLM（gemini-3-pro-for-verifier）
    ↓
LLM 响应: [MATCH FOUND] or [NO MATCH]
```

---

## 🎯 LLM 收到的具体信息

### 1. 系统上下文

```markdown
You are an expert in judging mathematical expression equivalence.
```

**作用**：告诉 LLM 它的角色

---

### 2. 任务说明

```markdown
## Task
Determine whether any of the following candidate formulas is 
**mathematically equivalent** to the Ground Truth (GT) formula after 
**appropriately choosing constant parameters**.
```

**作用**：清晰定义判断任务

---

### 3. 等价性规则（最关键）

```markdown
## Equivalence Rules
- The constants c0, c1, c2, ... in candidate formulas are undetermined 
  constant parameters that can take any real value (including 0).
  
- When a parameter equals 0, the corresponding term vanishes 
  (e.g., in `c0*sin(x) + c1*x`, setting c0=0 makes it equivalent to `c1*x`).
  
- Two formulas are equivalent if there exists an assignment of constants 
  such that the two formulas are equal for all values of the feature variables.
  
- Note algebraic identities, e.g., `A**2/(A**4*c0 + c1)` can be rewritten as 
  `1/(A**2*c0 + c1/A**2)`.
  
- Note exponent consolidation, e.g., in `c0*A**c1`, setting c1=2 makes it 
  equivalent to `c0*A**2`.
```

**作用**：定义什么是"等价"（这很重要，因为 LLM 需要理解参数可以是任意值）

---

### 4. 基础信息

```markdown
## Information
- **Feature variables**: {features}  # e.g., "t, P"
- **GT symbolic expression**: {gt_symbolic}  # e.g., "c0*(c1 - P/c2)*P + c3*P**c4"
```

**例子（对 BPG0）**：

```markdown
- **Feature variables**: t, P
- **GT symbolic expression**: c0*(c1 - P/c2)*P + c3*P**c4
```

**作用**：告诉 LLM 需要判断哪个 GT 表达式，以及涉及哪些变量

---

### 5. 候选公式列表（最长的部分）

```markdown
## Candidate Formula List
  1. P**2*c0*exp(c1*t) + P*c2*(P*c3)**c4 + P*c5*exp(c1*t) + P*c6
  2. P**2*c0 + P*c1 + P**c2*c3*exp(c4*t) + c5*t**c6 + c7
  3. P**(1/3)*c0 + P**2*c1 + P*c2 + c3*exp(P*c4)
  ...
  50. P**2*c0 + P*c1 + P**c2*c3  ← 最后的候选
  ...
  87. P*c0 + P*c1/(P*c2 + c3) + P**c4*c5 + c6/(P*c2 + c3)
```

**对 BPG0 而言**：87 个候选公式

**作用**：给 LLM 要判断的所有候选公式

---

### 6. 输出格式要求

```markdown
## Output Format
**You must output on the very first line** either `[MATCH FOUND]` or `[NO MATCH]`, 
then provide your analysis.
```

**作用**：规定 LLM 输出的格式，便于脚本解析

---

## 📊 以 BPG0 为例的完整提示内容

### Prompt 的结构

```
┌─────────────────────────────────────────────┐
│ You are an expert in judging mathematical   │
│ expression equivalence.                     │
├─────────────────────────────────────────────┤
│ ## Task                                     │
│ Determine whether any of the following...   │
├─────────────────────────────────────────────┤
│ ## Equivalence Rules                        │
│ - The constants c0, c1, c2, ... (5 条规则)  │
├─────────────────────────────────────────────┤
│ ## Information                              │
│ - **Feature variables**: t, P               │
│ - **GT**: c0*(c1 - P/c2)*P + c3*P**c4       │
├─────────────────────────────────────────────┤
│ ## Candidate Formula List                   │
│   1. P**2*c0*exp(c1*t) + ...                │
│   2. P**2*c0 + P*c1 + ...                   │
│   ...                                       │
│   50. P**2*c0 + P*c1 + P**c2*c3             │  ← 关键候选
│   ...                                       │
│   87. P*c0 + P*c1/(P*c2 + c3) + ...         │
├─────────────────────────────────────────────┤
│ ## Output Format                            │
│ **Must output on first line:**              │
│ `[MATCH FOUND]` or `[NO MATCH]`             │
└─────────────────────────────────────────────┘
```

**总字符数**：对 BPG0 约 10-15KB（取决于公式数量）

---

## 🔍 代码层面的实现

### 代码位置

[verify.py 第 107-152 行](verify.py#L107-L152) - `build_prompt()` 函数

```python
def build_prompt(gt_symbolic: str, gt_features: str, formulas: list[str]) -> str:
    features = gt_features.replace(";", ", ")
    formula_list = "\n".join(f"  {i+1}. {f}" for i, f in enumerate(formulas))

    return f"""You are an expert in judging mathematical expression equivalence.

## Task
Determine whether any of the following candidate formulas is **mathematically 
equivalent** to the Ground Truth (GT) formula after **appropriately choosing 
constant parameters**.

## Equivalence Rules
- The constants c0, c1, c2, ... in candidate formulas are undetermined 
  constant parameters that can take any real value (including 0).
- When a parameter equals 0, the corresponding term vanishes...
- Two formulas are equivalent if there exists an assignment of constants...
- Note algebraic identities...
- Note exponent consolidation...

## Information
- **Feature variables**: {features}
- **GT symbolic expression**: {gt_symbolic}

## Candidate Formula List
{formula_list}

## Output Format
**You must output on the very first line** either `[MATCH FOUND]` or 
`[NO MATCH]`, then provide your analysis."""
```

---

## 📈 信息提供的层级

| 层级 | 内容 | 例子 | 目的 |
|------|------|------|------|
| 1️⃣ **角色定位** | LLM 的身份 | "数学表达式等价性专家" | 定位 LLM 专业领域 |
| 2️⃣ **任务定义** | 具体做什么 | "判断候选公式是否等价于GT" | 明确任务目标 |
| 3️⃣ **规则说明** | 等价的定义 | "常数可任意设值包括0" <br> "项可消失" <br> "代数恒等式" | 在数学上规范问题 |
| 4️⃣ **上下文信息** | 变量和GT | "Feature: t, P" <br> "GT: c0*(c1 - P/c2)*P + c3*P**c4" | 提供具体数据 |
| 5️⃣ **候选列表** | 要判断的公式 | 87 个公式 | 提供选项 |
| 6️⃣ **输出格式** | 如何回答 | "[MATCH FOUND]" 或 "[NO MATCH]" | 便于结构化解析 |

---

## 💡 关键设计细节

### 为什么要给 LLM 这些信息？

1. **"常数可以是任意值"**
   - 允许 LLM 通过参数值的选择来建立等价性
   - 例：`P²*c0 + P*c1 + P**c2*c3` 
     - 可以通过 `c2=1/3` 变成 `P**1/3` 项
     - 从而与 GT 的 `P**c4` 部分匹配

2. **"项可以消失（参数=0）"**
   - 这是关键的灵活性
   - 允许候选公式有"多余的项"，只要这些项能被参数化为 0

3. **"代数恒等式"和"指数合并"**
   - 允许 LLM 进行数学推理
   - 不仅仅是字符串匹配

4. **完整的公式列表**
   - LLM 一次看到所有 87 个候选
   - 可以批量判断，而不是一个一个比较

---

## 🎯 LLM 的推理过程（以 Candidate 50 为例）

基于提供的信息，LLM 会这样推理：

```
GT: c0*(c1 - P/c2)*P + c3*P**c4
  = c0*c1*P - c0*P²/c2 + c3*P**c4
  = (c0*c1)*P + (-c0/c2)*P² + c3*P**c4
  = (常数)*P + (常数)*P² + c3*P**c4

↓ 观察结构

Candidate 50: P**2*c0 + P*c1 + P**c2*c3
  = c0*P² + c1*P + c3*P**c2

↓ 对比结构

相同！都是：
  (常数)*P² + (常数)*P + (常数)*P**(某指数)

↓ 结论

[MATCH FOUND] ✅
```

---

## 📊 总结表

| 信息类别 | 提供的具体内容 | 来源 | 用途 |
|--------|------------|------|------|
| **Ground Truth** | 符号表达式 | gt_expressions.csv | 判断目标 |
| **特征变量** | t, P 等 | gt_expressions.csv | 理解表达式的自由变量 |
| **等价性规则** | 5 条规则 | verify.py 硬编码 | 定义判断标准 |
| **候选公式** | 87 个 (for BPG0) | 搜索日志提取 | 判断对象 |
| **输出格式** | [MATCH FOUND] 或 [NO MATCH] | verify.py 硬编码 | 便于解析 |

---

## 🔗 相关代码位置

| 功能 | 文件位置 | 行号 |
|------|--------|------|
| 提取方程名 | [verify.py](verify.py#L14) | 14-19 |
| 加载 GT | [verify.py](verify.py#L22) | 22-28 |
| 提取公式 | [verify.py](verify.py#L40) | 40-77 |
| 构建提示 | [verify.py](verify.py#L107) | 107-152 |
| 调用 LLM | [verify.py](verify.py#L155) | 155-173 |
| 完整流程 | [verify.py](verify.py#L176) | 176-231 |

