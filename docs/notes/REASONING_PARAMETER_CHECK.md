# reasoning_effort 参数兼容性检查报告

## 🔴 关键问题：存在参数不兼容

### 问题描述
GPT-5.2 模型使用 `reasoning_effort` 时，存在严格的 API 约束：

**必须满足：**
- `temperature` 必须为 **1.0**（固定值，不能修改）

**禁止使用：**
- `top_p`
- `frequency_penalty`
- `presence_penalty`
- `top_k`

---

## 📋 检查结果

### 1️⃣ gpt-5-2.yaml 中的问题

#### ❌ selector 配置（第28-33行）
```yaml
selector:
  mode: requests
  model: gpt-5.2-2025-12-11
  base_url: https://llm-gateway.example.com/...
  api_key: REDACTED_API_KEY
  temperature: 0.3  # ⚠️ 问题：必须为 1.0
  max_tokens: 16384
  reasoning_effort: "low"  # 指定了 reasoning_effort
```

**问题：**
- 设置了 `reasoning_effort: "low"`，但 `temperature` 为 0.3 而非 1.0
- **性质：关键不兼容** — API 调用会被拒绝或强制覆盖

#### ❌ mutator 配置（第34-40行）
```yaml
mutator:
  mode: requests
  model: gpt-5.2-2025-12-11
  base_url: https://llm-gateway.example.com/...
  api_key: REDACTED_API_KEY
  temperature: 0.7  # ⚠️ 问题：必须为 1.0
  max_tokens: 16384
  reasoning_effort: "medium"  # 指定了 reasoning_effort
```

**问题：**
- 设置了 `reasoning_effort: "medium"`，但 `temperature` 为 0.7 而非 1.0
- **性质：关键不兼容** — API 调用会被拒绝或强制覆盖

#### ✅ generator 配置（第13-18行）
```yaml
generator:
  mode: requests
  model: gpt-5.2-2025-12-11
  ...
  max_tokens: 16384
  reasoning_effort: "medium"  # ← 指定了 reasoning_effort
```

**评估：**
- 没有显式设置 temperature（注释中有说明会自动设置为 1.0）
- **✓ 符合要求** — 系统会自动处理

#### ✅ 子组件配置
- `domain_knowledge`: temperature=0.7，**没有** reasoning_effort → ✓ 允许
- `seed_generation`: temperature=0.9，**没有** reasoning_effort → ✓ 允许
- `describe_batch`: temperature=0.5，**没有** reasoning_effort → ✓ 允许

---

### 2️⃣ 代码层面的处理

#### ✅ llm_client.py 中的保护代码（第108-122行）

```python
# ✅ Handle reasoning_effort parameter for advanced reasoning models
has_reasoning_effort = "reasoning_effort" in payload
if has_reasoning_effort and "gpt-5" in model.lower():
    # GPT-5.2 with reasoning_effort REQUIRES temperature=1.0
    payload["temperature"] = 1.0  # ← MUST be 1.0, no other values allowed
    # Remove incompatible parameters
    payload.pop("top_p", None)
    payload.pop("top_k", None)
    payload.pop("frequency_penalty", None)
    payload.pop("presence_penalty", None)
    print(f"[Reasoning Mode] GPT-5.2 reasoning_effort={payload['reasoning_effort']}, forced temperature=1.0")
```

**现状：**
- 代码会在运行时 **自动强制** `temperature=1.0`
- 代码会 **移除** 不兼容的参数（top_p, top_k, 等）
- ✓ 有日志输出提示

**问题：**
- 这是一种"事后补救"，不是"事前预防"
- YAML 配置中的错误参数值会被覆盖，可能导致困惑

---

### 3️⃣ 代码中加载配置的位置

#### main.py（第285-324行）— selector 加载
```python
sel_reasoning_effort = _resolve("selector", "reasoning_effort")
# ... 继承逻辑 ...
selector = create_selector(
    model=sel_model,
    base_url=sel_url, temperature=sel_temp,
    max_tokens=sel_tokens, usage_logger=usage_logger,
    llm_mode=sel_mode,
    anthropic_version=sel_anthropic_version,
    reasoning_effort=sel_reasoning_effort,
)
```

#### main.py（第327-351行）— mutator 加载
```python
mut_reasoning_effort = _resolve("mutator", "reasoning_effort")
# ... 继承逻辑 ...
return LLMMutator(
    api_client=client, model=mut_model,
    temperature=mut_temp, max_tokens=mut_tokens,
    usage_logger=usage_logger,
    anthropic_version=mut_anthropic_version,
    reasoning_effort=mut_reasoning_effort,
)
```

**现状：**
- temperature 和 reasoning_effort 都被传入
- 让底层 llm_client 处理兼容性

---

## 🔧 建议修复方案

### 方案 1：修正 YAML 配置（推荐）

**gpt-5-2.yaml** 应该改为：

```yaml
# selector 使用 reasoning_effort
selector:
  mode: requests
  model: gpt-5.2-2025-12-11
  base_url: https://llm-gateway.example.com/api/openai/deployments/gpt-5.2-2025-12-11/chat/completions?api-version=2024-05-01-preview
  api_key: REDACTED_API_KEY
  # ✅ 移除显式 temperature，让系统使用默认的 1.0
  # temperature: 1.0  # 可选注释，表示使用默认值
  max_tokens: 16384
  reasoning_effort: "low"

# mutator 使用 reasoning_effort
mutator:
  mode: requests
  model: gpt-5.2-2025-12-11
  base_url: https://llm-gateway.example.com/api/openai/deployments/gpt-5.2-2025-12-11/chat/completions?api-version=2024-05-01-preview
  api_key: REDACTED_API_KEY
  # ✅ 移除显式 temperature，让系统使用默认的 1.0
  # temperature: 1.0  # 可选注释，表示使用默认值
  max_tokens: 16384
  reasoning_effort: "medium"
```

**优势：**
- ✓ YAML 配置与 API 约束保持一致
- ✓ 避免配置值被运行时覆盖的混淆
- ✓ 显式表达设计意图

### 方案 2：增强代码提示和验证

在 main.py 中添加检查：

```python
def _validate_reasoning_config(component_name: str, cfg: dict):
    """验证 reasoning_effort 配置的兼容性"""
    if "reasoning_effort" in cfg:
        if "temperature" in cfg and cfg["temperature"] != 1.0:
            print(f"[WARNING] {component_name}: reasoning_effort specified with temperature={cfg['temperature']}")
            print(f"         ⚠️  temperature will be forced to 1.0 at runtime")
        if any(k in cfg for k in ["top_p", "frequency_penalty", "presence_penalty", "top_k"]):
            incompatible = [k for k in ["top_p", "frequency_penalty", "presence_penalty", "top_k"] if k in cfg]
            print(f"[WARNING] {component_name}: reasoning_effort with incompatible parameters: {incompatible}")
```

---

## 📊 完整问题清单

| 组件 | temperature | reasoning_effort | 状态 | 运行时处理 |
|------|------------|------------------|------|---------|
| generator | （默认1.0） | medium | ❌ 配置不规范 | ✓ 自动强制1.0 |
| domain_knowledge | 0.7 | 无 | ✓ OK | ✓ 无需处理 |
| seed_generation | 0.9 | 无 | ✓ OK | ✓ 无需处理 |
| describe_batch | 0.5 | 无 | ✓ OK | ✓ 无需处理 |
| selector | 0.3 | low | ❌ **不兼容** | ✓ 自动强制1.0 |
| mutator | 0.7 | medium | ❌ **不兼容** | ✓ 自动强制1.0 |

---

## ⚡ 行动项

### 必做：修正 YAML 配置

编辑 [gpt-5-2.yaml](gpt-5-2.yaml)：

1. 移除或注释掉 `selector` 中的 `temperature: 0.3`
2. 移除或注释掉 `mutator` 中的 `temperature: 0.7`
3. 保留 `reasoning_effort` 参数

**变更前后对比：**

```diff
  selector:
    mode: requests
    model: gpt-5.2-2025-12-11
    base_url: https://llm-gateway.example.com/...
    api_key: REDACTED_API_KEY
-   temperature: 0.3
+   # temperature: 1.0  # ← reasoning_effort 使用时固定为 1.0
    max_tokens: 16384
    reasoning_effort: "low"

  mutator:
    mode: requests
    model: gpt-5.2-2025-12-11
    base_url: https://llm-gateway.example.com/...
    api_key: REDACTED_API_KEY
-   temperature: 0.7
+   # temperature: 1.0  # ← reasoning_effort 使用时固定为 1.0
    max_tokens: 16384
    reasoning_effort: "medium"
```

### 可选：增强代码验证

在 [main.py](main.py#L285) 中添加配置验证函数

---

## 参考文献

- GPT-5.2 API 文档：reasoning_effort 模式限制
- [llm_client.py](src/llm_client.py#L108) 中的处理逻辑
- [gpt-5-2.yaml](gpt-5-2.yaml) 配置文件

