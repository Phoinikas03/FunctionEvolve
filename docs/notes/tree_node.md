# 搜索树节点摘要说明

本文整理 `symregression` 里搜索树 `Database` 的节点结构，重点说明：

1. 搜索树里每个节点 `EvolutionNode` 实际保存了哪些信息
2. 对外暴露给搜索流程的“节点摘要”包含哪些条目
3. 这些摘要条目是在哪里生成、给谁用的

核心代码位置：

- [`src/evolution_tree.py`](../src/evolution_tree.py)
- [`src/search.py`](../src/search.py)
- [`src/selector.py`](../src/selector.py)

## 1. `Database` 里存的是什么

在这套代码里，图中 `Database` 对应的是 `EvolutionTree`。

`EvolutionTree` 维护的是**整个搜索树**，而不是简单的候选列表。它主要保存三类结构：

- `_nodes: Dict[str, EvolutionNode]`
  - 键是公式字符串
  - 值是对应的节点对象
- `_children: Dict[str, List[str]]`
  - 记录父公式到子公式的邻接关系
- `_roots: List[str]`
  - 记录初始 seed 节点

对应实现见 [`src/evolution_tree.py:184-188`](../src/evolution_tree.py#L184)。

所以从数据结构上看，`Database` 里真正保存的是：

- 整棵树的节点对象
- 节点之间的父子关系
- 根节点集合

## 2. 单个节点 `EvolutionNode` 保存的完整信息

单个节点定义在 [`src/evolution_tree.py:6-169`](../src/evolution_tree.py#L6)。

一个节点的完整字段可以按下面几类理解。

### 2.1 基本身份信息

- `skeleton_str`
  - 公式字符串，是节点的主键
- `parent_id`
  - 父节点公式字符串；根节点时可能是 `None`

对应代码：[`src/evolution_tree.py:13-15`](../src/evolution_tree.py#L13)

### 2.2 评估分数信息

- `train_nmse`
- `test_nmse`
- `ood_test_nmse`
- `is_evaluated`

这些字段表示该公式经过参数拟合后，在不同数据划分上的误差情况，以及是否已经完成评估。

对应代码：[`src/evolution_tree.py:16-19`](../src/evolution_tree.py#L16)

### 2.3 AST / 结构信息

- `ast_features`
  - AST 特征集合，例如不同操作符和深度信息
- `operator_counts`
  - 各类操作符计数
- `sympy_expr`
  - SymPy 解析后的表达式对象
- `parse_error`
  - 表达式解析是否失败

这些信息在节点创建时就会自动解析并提取。

对应代码：

- 字段定义：[`src/evolution_tree.py:21-24`](../src/evolution_tree.py#L21)
- AST 提取：[`src/evolution_tree.py:48-72`](../src/evolution_tree.py#L48)

### 2.4 参数信息

- `param_names`
  - 常数参数名列表，例如 `["c0", "c1"]`
- `fitted_params`
  - 拟合后的常数值列表

对应代码：[`src/evolution_tree.py:26-28`](../src/evolution_tree.py#L26)

### 2.5 搜索状态标记

- `is_mature`
  - 是否达到“成熟节点”阈值
- `is_degenerated`
  - 是否被判定为退化 / 过拟合 / 可化简退化节点

对应代码：[`src/evolution_tree.py:30-31`](../src/evolution_tree.py#L30)

### 2.6 描述信息

- `_rule_description`
  - 基于 AST 自动生成的规则描述
- `_llm_description`
  - 后续由 Generator 生成的 LLM 描述
- `description`
  - 对外统一使用的描述属性
  - 优先返回 `_llm_description`，否则回退到 `_rule_description`

对应代码：

- 字段：[`src/evolution_tree.py:33-35`](../src/evolution_tree.py#L33)
- 属性：[`src/evolution_tree.py:40-46`](../src/evolution_tree.py#L40)
- 规则描述生成：[`src/evolution_tree.py:74-123`](../src/evolution_tree.py#L74)

## 3. 什么叫“节点摘要”

虽然节点里保存了很多完整信息，但真正对外用于搜索决策的摘要，不是整颗 `EvolutionNode` 原样暴露，而是通过：

- `EvolutionNode.to_summary_dict()`

生成一个较轻量的字典。

对应实现见 [`src/evolution_tree.py:148-165`](../src/evolution_tree.py#L148)。

这个摘要是 `Selector` 在每一轮做父节点选择时看到的内容，也是“Database 中节点摘要信息”的最直接来源。

## 4. 节点摘要条目清单

下面是 `to_summary_dict()` 当前实际生成的摘要字段。

### `id`

- 含义：节点唯一标识
- 值：公式字符串本身
- 来源：`self.skeleton_str`

代码：[`src/evolution_tree.py:152`](../src/evolution_tree.py#L152)

### `formula`

- 含义：公式文本
- 值：公式字符串本身
- 与 `id` 的关系：当前实现里和 `id` 相同

代码：[`src/evolution_tree.py:153`](../src/evolution_tree.py#L153)

### `description`

- 含义：节点的结构描述
- 优先级：
  - 有 LLM 描述时，用 LLM 描述
  - 否则回退到规则生成描述

代码：[`src/evolution_tree.py:154`](../src/evolution_tree.py#L154)

### `train_nmse`

- 含义：训练集归一化误差
- 特点：
  - 只有在节点已评估且数值正常时才填入
  - 否则为 `None`
  - 会被 `round(..., 6)` 处理

代码：[`src/evolution_tree.py:155`](../src/evolution_tree.py#L155)

注意：摘要里**只暴露 `train_nmse`，不暴露 `test_nmse` 和 `ood_test_nmse`**。  
这是刻意设计，用来避免 Selector 直接基于测试集信息做决策。

### `parent_id`

- 含义：父节点公式 ID
- 用途：帮助理解该节点在树中的来源

代码：[`src/evolution_tree.py:156`](../src/evolution_tree.py#L156)

### `n_children`

- 含义：这个节点当前已经扩展出的孩子数量
- 来源：不是节点内部字段，而是 `to_summary_dict(n_children=...)` 时由树结构传入

代码：

- 字段写入：[`src/evolution_tree.py:157`](../src/evolution_tree.py#L157)
- 调用位置：[`src/evolution_tree.py:256-258`](../src/evolution_tree.py#L256)

### `n_params`

- 含义：该公式包含的可调常数参数个数
- 值：`len(self.param_names)`

代码：[`src/evolution_tree.py:158`](../src/evolution_tree.py#L158)

### `depth`

- 含义：AST 树深度
- 来源：`self.tree_depth`

代码：[`src/evolution_tree.py:159`](../src/evolution_tree.py#L159)

### `n_operators`

- 含义：AST 中操作符总数
- 来源：`sum(self.operator_counts.values())`

代码：[`src/evolution_tree.py:160`](../src/evolution_tree.py#L160)

### `fitted_params`

- 含义：拟合后的参数字符串
- 格式示例：`c0=1.2345e+00, c1=-3.2000e-01`
- 特点：
  - 只有存在已拟合参数时才写入
  - 否则这个字段不会出现在摘要字典里

代码：

- 条件添加：[`src/evolution_tree.py:162-164`](../src/evolution_tree.py#L162)
- 格式化函数：[`src/evolution_tree.py:139-146`](../src/evolution_tree.py#L139)

## 5. 摘要条目总表

| 字段名 | 类型 | 含义 | 是否总是出现 |
| --- | --- | --- | --- |
| `id` | `str` | 节点唯一标识，当前就是公式字符串 | 是 |
| `formula` | `str` | 公式字符串 | 是 |
| `description` | `str` | 节点结构描述 | 是 |
| `train_nmse` | `float \| None` | 训练集 NMSE | 是 |
| `parent_id` | `str \| None` | 父节点公式 ID | 是 |
| `n_children` | `int` | 当前孩子数 | 是 |
| `n_params` | `int` | 参数个数 | 是 |
| `depth` | `int` | AST 深度 | 是 |
| `n_operators` | `int` | 操作符总数 | 是 |
| `fitted_params` | `str` | 拟合参数串 | 否，只有有参数时出现 |

## 6. 哪些信息在节点里有，但默认不在摘要里

下面这些信息虽然保存在 `EvolutionNode` 里，但**不会默认出现在 `to_summary_dict()` 的摘要中**：

- `test_nmse`
- `ood_test_nmse`
- `is_evaluated`
- `is_mature`
- `is_degenerated`
- `ast_features`
- `operator_counts`
- `sympy_expr`
- `parse_error`
- `param_names`
- `fitted_params` 原始数值列表

这些信息更多是：

- 给搜索流程内部使用
- 给日志 / 最终结果使用
- 给退化检测、成熟节点判断使用
- 或给更底层的结构分析使用

其中比较重要的几点：

### `test_nmse` / `ood_test_nmse`

节点里有，但摘要里不放。  
这样做是为了减少测试集信息泄漏到 Selector 决策过程。

### `is_mature` / `is_degenerated`

它们也不直接进摘要，但会影响 `get_tree_summary()` 是否把节点纳入候选池。

例如在 [`src/evolution_tree.py:247-255`](../src/evolution_tree.py#L247)：

- `is_degenerated=True` 的节点会被过滤掉
- 如果 `exclude_mature=True`，成熟节点也会被过滤掉

## 7. 摘要是怎么被生成出来的

摘要的生成链路是：

1. `EvolutionTree.get_tree_summary(...)`
2. 遍历符合条件的节点
3. 调用 `node.to_summary_dict(n_children=...)`
4. 返回摘要列表

代码位置：[`src/evolution_tree.py:238-259`](../src/evolution_tree.py#L238)

这里还有几个重要规则：

- 先取已评估节点，再按 `train_nmse` 升序排序
- 未评估节点会拼接到后面
- 最多返回 `max_nodes`
- 始终排除 `is_degenerated` 节点
- 可选排除 `is_mature` 节点

所以“摘要列表”不仅是字段集合，还带有一层筛选和排序语义。

## 8. 这些摘要给谁用

最直接的消费者是 `Selector`。

在 [`src/search.py:397-402`](../src/search.py#L397) 中：

```python
plans = self.selector.plan(
    tree_summary=summary,
    context_prompt=self.dataset.get_context_prompt(),
    candidate_num=self.candidate_num,
    selection_history=self._selection_history,
)
```

也就是说：

- `Database` 先产出树摘要 `summary`
- `Selector` 再基于这些摘要挑出下一轮父节点

从 `selector.py` 的 prompt 说明也能看出来，Selector 预期看到的摘要字段就是：

- `id`
- `formula`
- `description`
- `train_nmse`
- `n_children`
- `n_params`
- `depth`
- `n_operators`
- `fitted_params`

见 [`src/selector.py:32-42`](../src/selector.py#L32)。

## 9. 一个摘要样例

按当前实现，一个典型节点摘要大致会长这样：

```python
{
    "id": "c0*x + c1",
    "formula": "c0*x + c1",
    "description": "top-level additive combination, contains powers, simple structure",
    "train_nmse": 0.001234,
    "parent_id": "c0*x",
    "n_children": 3,
    "n_params": 2,
    "depth": 2,
    "n_operators": 5,
    "fitted_params": "c0=1.2030e+00, c1=-2.1000e-01"
}
```

注意这只是摘要视图，不是节点完整内部状态。

## 10. 一句话总结

如果只抓最关键的一点：

- `EvolutionNode` 保存的是**完整节点状态**
- `to_summary_dict()` 暴露的是**供 Selector 和搜索调度使用的轻量摘要**

当前摘要的核心条目就是：

`id / formula / description / train_nmse / parent_id / n_children / n_params / depth / n_operators / fitted_params`
