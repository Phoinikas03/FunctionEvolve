# reasoning_effort 参数使用指南

## 概述
`reasoning_effort` 是 GPT-5.2 等高级 LLM 模型支持的参数，用于控制模型的推理深度和计算资源投入。

## 参数值
- `"low"` - 低推理资源投入，速度快，成本低，适合简单任务
- `"medium"` - 中等推理资源投入，平衡性能和成本（推荐）
- `"high"` - 高推理资源投入，推理深度最大，成本最高，适合复杂任务

## 使用方式

### 1. 在 YAML 配置文件中设置

#### 为所有组件统一设置（在 gpt-5-2.yaml）
```yaml
generator:
  mode: requests
  model: gpt-5.2-2025-12-11
  base_url: https://llm-gateway.example.com/api/openai/deployments/gpt-5.2-2025-12-11/chat/completions?api-version=2024-05-01-preview
  api_key: your-api-key
  temperature: 0.8
  max_tokens: 16384
  reasoning_effort: "medium"  # ← 统一设置推理深度
```

#### 为特定组件单独设置
```yaml
generator:
  mode: requests
  model: gpt-5.2-2025-12-11
  base_url: https://llm-gateway.example.com/api/openai/deployments/gpt-5.2-2025-12-11/chat/completions?api-version=2024-05-01-preview
  api_key: your-api-key
  temperature: 0.8
  max_tokens: 16384
  reasoning_effort: "medium"

# 域知识生成使用更高的推理深度
domain_knowledge:
  reasoning_effort: "high"

# 种子生成使用默认推理深度
seed_generation:
  temperature: 0.9
  max_tokens: 8192

# 选择器使用较低的推理深度
selector:
  mode: requests
  model: gpt-5.2-2025-12-11
  base_url: https://llm-gateway.example.com/api/openai/deployments/gpt-5.2-2025-12-11/chat/completions?api-version=2024-05-01-preview
  api_key: your-api-key
  temperature: 0.3
  max_tokens: 16384
  reasoning_effort: "low"

# 变异器使用中等推理深度
mutator:
  mode: requests
  model: gpt-5.2-2025-12-11
  base_url: https://llm-gateway.example.com/api/openai/deployments/gpt-5.2-2025-12-11/chat/completions?api-version=2024-05-01-preview
  api_key: your-api-key
  temperature: 0.7
  max_tokens: 16384
  reasoning_effort: "medium"
```

### 2. 在 Python 代码中直接使用

```python
from src.llm_client import build_openai_client

# 建立客户端
url_gpt52 = "https://llm-gateway.example.com/api/openai/deployments/gpt-5.2-2025-12-11/chat/completions?api-version=2024-05-01-preview"
client = build_openai_client(mode="requests", model="gpt-5.2-2025-12-11", base_url=url_gpt52)

# 使用 reasoning_effort 参数调用 API
response = client.chat.completions.create(
    model="gpt-5.2-2025-12-11",
    messages=[{"role": "user", "content": "Your prompt here"}],
    reasoning_effort="medium",     # ← 设置推理深度
    max_tokens=2048
)

content = response.choices[0].message.content
print(content)
```

## 运行示例

### 测试脚本
```bash
python test_api.py
```

针对 GPT-5.2 的测试代码已包含 `reasoning_effort="medium"` 设置。

### 使用自定义配置运行搜索
```bash
python main.py --llm-config gpt-5-2.yaml --dataset bio_pop_growth
```

配置文件中的 `reasoning_effort` 会自动传递给所有 LLM 组件。

## 智能继承机制

系统实现了智能继承机制：
- 如果 `generator` 组件设置了 `reasoning_effort`
- 而 `domain_knowledge`, `seed_generation`, `describe_batch` 没有单独设置
- 且它们使用相同的 model/base_url
- **那么这些子组件会自动继承** generator 的 `reasoning_effort` 设置

这样可以避免重复配置，但仍然允许为特定组件覆盖。

## 推荐配置

### 用于复杂符号回归问题 (GPT-5.2 with reasoning)
```yaml
generator:
  mode: requests
  model: gpt-5.2-2025-12-11
  base_url: https://llm-gateway.example.com/api/openai/deployments/gpt-5.2-2025-12-11/chat/completions?api-version=2024-05-01-preview
  api_key: your-api-key
  # ⚠️ 注意：temperature 将被强制设为 1.0
  max_tokens: 4096
  reasoning_effort: "high"      # 需要深度分析的初始知识生成

domain_knowledge:
  reasoning_effort: "high"      # 深度分析科学领域

seed_generation:
  reasoning_effort: "medium"    # 平衡创意和成本

selector:
  reasoning_effort: "low"       # 简单排名不需要高深度

mutator:
  reasoning_effort: "medium"    # 变异建议需要适度推理
```

### 用于快速原型验证 (GPT-5.2 with reasoning)
```yaml
generator:
  reasoning_effort: "low"       # 快速生成初步解

domain_knowledge:
  reasoning_effort: "low"       # 快速获取基本知识

seed_generation:
  reasoning_effort: "low"       # 快速种子生成

selector:
  reasoning_effort: "low"       # 快速排序

mutator:
  reasoning_effort: "low"       # 快速变异建议
```

## 技术细节

### 参数传递链
1. **YAML 配置** → 加载到 main.py
2. **main.py** → 创建 component_overrides
3. **component_overrides** → 传递给 LLMGenerator/Selector/LLMMutator
4. **各个组件** → 在 API 调用时传递给 LLM
5. **llm_client.py** → 处理特殊参数转换（如 max_tokens → max_completion_tokens）

### API 兼容性要求

#### temperature 参数强制要求 ⚠️
使用 `reasoning_effort` 时，**temperature 必须为 1.0（固定值，不可自定义）**：

```python
# ✅ 正确 - 不指定 temperature（系统自动设为 1.0）
response = client.chat.completions.create(
    model="gpt-5.2-2025-12-11",
    messages=[...],
    reasoning_effort="medium",
    max_tokens=2048
)

# ✅ 也正确 - 显式设置为 1.0
response = client.chat.completions.create(
    model="gpt-5.2-2025-12-11",
    messages=[...],
    reasoning_effort="medium",
    temperature=1.0,  # ← 必须为 1.0
    max_tokens=2048
)

# ❌ 错误 - temperature != 1.0 会报错
response = client.chat.completions.create(
    model="gpt-5.2-2025-12-11",
    messages=[...],
    reasoning_effort="medium",
    temperature=0.5,  # ✗ 不允许！
    max_tokens=2048
)
```

**系统会自动处理：**
- 如果使用 `reasoning_effort`，自动强制设置 `temperature=1.0`
- 移除所有不兼容参数：`top_p`、`top_k`、`frequency_penalty`、`presence_penalty`
- 打印 `[Reasoning Mode]` 日志表示参数已调整

#### 参数互斥性
- `reasoning_effort` 与以下参数**不兼容**：
  - `top_p` （自动移除）
  - `top_k` （自动移除）
  - `frequency_penalty` （自动移除）
  - `presence_penalty` （自动移除）
  - `temperature != 1.0` （自动调整为 1.0）

#### 已知兼容性问题

| 问题 | 原因 | 解决方案 |
|------|------|--------|
| 400 Bad Request: temperature 不支持 0.5 | temperature 必须为 1.0 | 移除 temperature 参数或设为 1.0 |
| 响应为空 | max_completion_tokens 太小或参数冲突 | 增加 max_tokens 到 >= 2048 |
| API 参数验证失败 | top_p/frequency_penalty 与 reasoning_effort 冲突 | 系统自动移除，或手动不指定这些参数 |
| 处理超时 | reasoning_effort="high" 消耗过多资源 | 改用 "low" 或 "medium"，或增加超时时间 |
| 成本增加 | 推理深度越高，token 消耗越多 | 根据实际需求调整 effort 级别 |

## 常见问题

**Q: 400 错误："temperature does not support 0.5 with this model"**
A: 这是 GPT-5.2 o1 系列的强制要求：
- **temperature 必须为 1.0（唯一可用值）**
- 不能设置为任何其他值（包括 0.3、0.5、0.7 等）
- 推理模型不支持自定义 temperature

解决方案：
```yaml
# ❌ 错误
generator:
  reasoning_effort: "medium"
  temperature: 0.5              # 会报错

# ✅ 正确
generator:
  reasoning_effort: "medium"
  # 不设置 temperature（系统自动设为 1.0）
  # 或显式设为 1.0
  temperature: 1.0
```

**Q: 为什么 GPT-5.2 使用 reasoning_effort 后没有输出？**
A: 最常见的原因和解决方案：
1. **temperature 设置错误** → 改为 1.0（或不设置）
2. **max_tokens 太小** → 增加到 >= 2048
3. **包含不兼容参数** → 系统会自动移除 top_p、frequency_penalty 等
4. 查看日志中的 `[Reasoning Mode]` 消息确认参数已调整

**Q: 能否在不同组件使用不同的 reasoning_effort？**
A: 可以。每个组件都可以独立设置：
```yaml
generator:
  reasoning_effort: "high"      # 深度推理

domain_knowledge:
  reasoning_effort: "high"      # 继承或独自设置

selector:
  reasoning_effort: "low"       # 简单排序

mutator:
  reasoning_effort: "medium"    # 中等推理
```

**Q: reasoning_effort 会增加成本吗？**
A: 
- **是的，会显著增加成本**
- `"high"` > `"medium"` > `"low"`
- 建议根据任务复杂度调整
- 简单任务用 "low"，复杂任务用 "high"

**Q: 如何调试 reasoning_effort 参数问题？**
A: 
1. 查看 `[Reasoning Mode]` 日志：确认参数已调整
2. 查看 `[API Error]` 日志：显示具体错误
3. 检查参数：temperature=1.0、max_tokens 足够大
4. 逐个调整 reasoning_effort 级别（low → medium → high）
