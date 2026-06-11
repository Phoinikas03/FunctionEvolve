# `main.py` 逻辑拆解：以 Generator、Selector、Mutator、Optimizer 为核心

本文聚焦 [`main.py`](../main.py) 的主流程，并把代码中的 4 个核心组件串起来讲清楚：

1. `Generator`：提供领域先验、初始种子、结构描述
2. `Selector`：在当前进化树里挑选下一轮父节点
3. `Mutator`：围绕父节点生成新结构，并做去重/退化处理
4. `Optimizer`：对候选结构里的常数参数做拟合，产出 `train/test/ood` 误差

虽然 `main.py` 是入口，但真正的“搜索引擎”主要落在 [`src/search.py`](../src/search.py)。`main.py` 的核心职责更像是：

- 读取数据集和配置
- 装配 4 个核心组件
- 把组件交给 `TreeSearch`
- 启动搜索并输出最终结果

## 1. 主流程总览

`run()` 定义在 [`main.py:163`](../main.py#L163)。主流程可以概括为下面 7 步：

1. 读取数据集，构建 `SRDataset`（[`main.py:236-247`](../main.py#L236)）
2. 创建日志路径和 usage 统计路径（[`main.py:249-267`](../main.py#L249)）
3. 创建 `Evaluator`，并指定常数优化器 `optimizer`（[`main.py:270-280`](../main.py#L270)）
4. 根据配置装配 `Generator`、`Selector`、`LLMMutator`（[`main.py:347-475`](../main.py#L347)）
5. 用这些组件创建 `TreeSearch`（[`main.py:486-511`](../main.py#L486)）
6. 先做 `initialize_seeds()`，再做 `run()` 正式搜索（[`main.py:578-580`](../main.py#L578)）
7. 读取全局最优结果，输出日志，并做等价验证（[`main.py:582-620`](../main.py#L582)）

如果只看职责分工，可以把主链路理解成：

```text
dataset/config
   -> Generator / Selector / Mutator / Evaluator(Optimizer)
   -> TreeSearch.initialize_seeds()
   -> TreeSearch.run()
   -> best result
```

## 2. 四个核心组件在主流程中的位置

### 2.1 Generator：负责“起点”和“解释”

`main.py` 中的装配位置在 [`main.py:347-409`](../main.py#L347)。

它不是只做“生成公式”这一件事，而是拆成了 3 类子职责，全部收拢在 [`src/generator.py`](../src/generator.py) 的 `LLMGenerator`：

- `generate_domain_knowledge()`：先根据任务背景生成领域知识、启发式和可选预处理规则（[`src/generator.py:222-345`](../src/generator.py#L222)）
- `initialize_seeds()`：基于领域知识生成初始种子公式（[`src/generator.py:346-399`](../src/generator.py#L346)）
- `describe()` / `describe_batch()`：在候选公式已经拟合完参数后，为公式补充自然语言结构描述（[`src/generator.py:400-483`](../src/generator.py#L400)）

### Generator 在搜索里的真实调用时机

Generator 不是只在最开始调用一次，而是贯穿前后两个阶段：

#### 阶段 A：初始化种群

在 [`src/search.py:223-357`](../src/search.py#L223) 的 `initialize_seeds()` 里：

1. 先调用 `generator.generate_domain_knowledge()`
2. 把生成出来的领域知识同步给 `llm_mutator.domain_knowledge`（[`src/search.py:234-236`](../src/search.py#L234)）
3. 再调用 `generator.initialize_seeds()` 生成初始公式
4. 初始种子经过评估后，再用 `generator.describe_batch()` 给有效种子补结构描述

所以 Generator 既决定“从哪里出发”，也决定“树里的节点如何被描述”。

#### 阶段 B：搜索中给优秀子节点补描述

在正式搜索中，如果某个 child 的 `test_nmse` 比 parent 更好，会再调用一次 `generator.describe_batch()`（[`src/search.py:443-465`](../src/search.py#L443)）。

这意味着 Generator 还承担了“为后续 Selector 提供可读上下文”的作用，因为 `EvolutionNode.description` 会进入 Selector 的输入摘要。

### Generator 的退化模式

如果 `degenerated_generator=True`，`main.py` 会改用 `MockGenerator`（[`main.py:350-353`](../main.py#L350)）：

- 不调用 LLM
- 领域知识为空
- 种子来自内置 fallback 列表（[`src/generator.py:577-602`](../src/generator.py#L577)）
- 描述为空

可以把它理解为“把搜索起点固定死，只保留后面的搜索机制”。

## 3. Selector：负责“下一步扩展谁”

`main.py` 中的装配位置在 [`main.py:411-434`](../main.py#L411)。

对应实现主要在 [`src/selector.py`](../src/selector.py)。

### Selector 的输入是什么

在每一轮搜索开始时，`TreeSearch.run()` 会从 `EvolutionTree` 中取一个摘要列表 `summary`（[`src/search.py:382-392`](../src/search.py#L382)），然后调用：

```python
plans = self.selector.plan(
    tree_summary=summary,
    context_prompt=self.dataset.get_context_prompt(),
    candidate_num=self.candidate_num,
    selection_history=self._selection_history,
)
```

也就是 [`src/search.py:397-402`](../src/search.py#L397)。

这个 `tree_summary` 来自 [`src/evolution_tree.py:238-259`](../src/evolution_tree.py#L238)，每个节点摘要主要包含：

- `id/formula`
- `description`
- `train_nmse`
- `n_children`
- `n_params`
- `depth`
- `n_operators`
- `fitted_params`

一个很关键的设计是：摘要里**只暴露 `train_nmse`，不把 `test_nmse` 直接喂给 Selector**（见 [`src/evolution_tree.py:148-165`](../src/evolution_tree.py#L148) 的注释和实现）。这样可以减少搜索时直接朝测试集“刷分”的风险。

### Selector 的职责

`SelectorLLMAgent.plan()` 定义在 [`src/selector.py:144-208`](../src/selector.py#L144)。它做的事情是：

1. 把当前树摘要格式化成 prompt
2. 把历史父节点选择记录也一起给模型
3. 让模型选出 `candidate_num` 个父节点
4. 返回形如 `{"parent_id": ..., "rationale": ...}` 的列表

Selector 本身不生成新公式，它只决定“把预算投给哪些方向”。

### Selector 在整体策略里的价值

它并不是单纯挑当前最优节点，而是做带探索性的路由：

- 倾向低误差但结构还比较简单的节点
- 避免一直重复扩展同一类结构
- 参考历史选择记录，防止某个方向被过度开发

所以它在系统里更像“搜索调度器”，而不是评分器。

### Selector 的两种退化模式

#### `degenerated_selector1`

使用 `MockSelector`（[`main.py:412-415`](../main.py#L412)），基于 rank + Boltzmann 采样选父节点（[`src/selector.py:318-396`](../src/selector.py#L318)）。

#### `degenerated_selector2`

仍然使用 LLM Selector，但 prompt 中剥离 AST 相关字段和描述字段（[`main.py:431-434`](../main.py#L431)，[`src/selector.py:156-167`](../src/selector.py#L156)）。

这相当于保留“LLM 做决策”，但削弱它可见的结构信息。

## 4. Mutator：负责“如何从父节点长出孩子”

在 `main.py` 里，Mutator 实际上分成两层：

- `self.mutator = ASTMutator(...)`：程序化结构变异器，在 `TreeSearch.__init__()` 内部直接创建（[`src/search.py:203`](../src/search.py#L203)）
- `llm_mutator = LLMMutator(...)`：LLM 驱动的结构建议器，由 `main.py` 装配（[`main.py:436-475`](../main.py#L436)）

因此“Mutator”在实现上不是一个类，而是一个组合：

1. AST 程序变异
2. LLM 结构建议
3. 统一去重、退化检测、必要时化简重评估

### 4.1 ASTMutator：程序化变异骨架

核心实现在 [`src/mutator.py`](../src/mutator.py)。

它承担 4 类关键工作：

#### 1. 公式规范化与结构去重

相关方法：

- `normalize_expression()`（[`src/mutator.py:68-90`](../src/mutator.py#L68)）
- `structural_key()`（[`src/mutator.py:147-157`](../src/mutator.py#L147)）

作用是把表达式整理到更稳定的规范形式，再提取一个“忽略参数命名差异”的结构指纹，便于全局去重。

这件事很关键，因为搜索会不断地产生语法不同但结构等价的公式；没有这层，树会很快充满重复分支。

#### 2. 程序化删除变异

`enumerate_deletions()` 定义在 [`src/mutator.py:510-555`](../src/mutator.py#L510)。

典型操作包括：

- 删除 `Add` 里的某一项
- 删除 `Mul` 里的某个因子
- 去掉函数包裹，如 `exp(f) -> f`
- 去掉幂次，如 `x**n -> x`

可以把它看成“做简化、做剪枝”的方向。

#### 3. 程序化添加变异

`enumerate_additions()` 定义在 [`src/mutator.py:601-726`](../src/mutator.py#L601)。

它会枚举一批模板化扩展项，例如：

- 线性项
- 正弦项
- 幂函数项
- 指数项
- 对数项
- 有理式项
- 幂律耦合项

可以把它看成“在当前公式外面再长一层结构”的方向。

#### 4. 退化检测与参数后处理

相关方法：

- `check_degeneracy()`（[`src/mutator.py:761-827`](../src/mutator.py#L761)）
- `normalize_fitted_params()`（[`src/mutator.py:968`](../src/mutator.py#L968)）

它会在参数拟合之后检查：

- 是否出现过大系数，提示复杂结构在硬凑数据
- 是否有小参数把某些子结构实际压扁成 0 或更简单形式
- 是否能把退化后的结构进一步化简成新公式

如果化简出了一个更简单的新结构，`TreeSearch` 会把这个化简结果重新加入评估队列（[`src/search.py:839-879`](../src/search.py#L839)）。

这一步很像“搜索过程中的自动蒸馏”。

### 4.2 LLMMutator：补充程序模板覆盖不到的变异

`LLMMutator.suggest_mutations()` 定义在 [`src/mutator.py:1478-1542`](../src/mutator.py#L1478)。

它的输入包括：

- 当前任务背景
- 父公式
- 父公式 AST 文本
- 已拟合参数
- 程序化变异已经做过哪些事
- 历史 top 表达式
- toxic motifs

然后返回最多 20 个结构建议。

这说明 LLM Mutator 的定位不是“随便乱改”，而是：

- 看懂父节点当前结构
- 避开程序规则已经覆盖的简单修改
- 借助领域知识提出更跳跃、更启发式的结构变化

### 4.3 `TreeSearch` 如何把两类变异拼起来

候选生成入口是 [`src/search.py:924-980`](../src/search.py#L924) 的 `_generate_candidates()`。

顺序如下：

1. 如果没关闭程序化变异，先做 `enumerate_deletions()` 和 `enumerate_additions()`
2. 再调用 `llm_mutator.suggest_mutations()`
3. 对 LLM 结果再做一次 `normalize_expression()`
4. 合并三路结果
5. 用 `structural_key + _seen_keys` 去重

所以 Mutator 不是单一路径，而是“模板搜索 + LLM 搜索”的并联结构。

### 4.4 Mutator 的退化模式

`main.py` 里有 3 个 Mutator 相关开关：

- `degenerated_mutator1`：只保留程序化变异，不调用 LLM（[`main.py:457-460`](../main.py#L457)）
- `degenerated_mutator2`：只保留 LLM 变异，跳过程序化变异（[`main.py:468-470`](../main.py#L468)）
- `degenerated_mutator3`：和 `mutator2` 类似，但 LLM prompt 里还会去掉 AST-heavy 信息（[`main.py:461-467`](../main.py#L461)）

这几个开关本质上是在做消融实验，帮助区分：

- AST 模板本身有多大贡献
- LLM 结构建议有多大贡献
- AST 可见性对 LLM 变异质量有多大贡献

## 5. Optimizer：负责“同一结构下把常数拟合到最好”

`Optimizer` 在 `main.py` 里不是独立直接调用，而是通过 `Evaluator` 间接使用。

装配点在 [`main.py:270-280`](../main.py#L270)：

```python
evaluator = Evaluator(..., optimizer=optimizer)
```

### 5.1 Evaluator 和 Optimizer 的关系

`Evaluator` 定义在 [`src/evaluator.py`](../src/evaluator.py)。

它本身不实现优化算法，只做两件事：

1. 调用 `optimizer.optimize()` 拟合常数参数（[`src/evaluator.py:103-116`](../src/evaluator.py#L103)）
2. 用拟合后的参数计算 `train/test/ood_test` 的 NMSE（[`src/evaluator.py:124-142`](../src/evaluator.py#L124)）

也就是说：

- `Optimizer` 解决的是“参数怎么找”
- `Evaluator` 解决的是“找完参数后怎么统一评估”

### 5.2 Optimizer 是如何选择实现的

在 [`src/optimizer/__init__.py:18-35`](../src/optimizer/__init__.py#L18) 里注册了几种优化器：

- `DE`
- `CMA-ES`
- `L-BFGS-B`
- `least_squares`
- `Structure`

`Evaluator` 在初始化时通过 `get_optimizer(optimizer, **kwargs)` 创建具体优化器（[`src/evaluator.py:69-70`](../src/evaluator.py#L69)）。

### 5.3 默认 `Structure` 优化器做了什么

默认值是 `Structure`（[`main.py:178`](../main.py#L178)），实现位于 [`src/optimizer/structure.py:23-246`](../src/optimizer/structure.py#L23)。

它不是单一算法，而是一个分阶段策略，大致是：

1. 先分析表达式的参数约束、幂指数、正值约束等
2. 做一轮和幂函数相关的预搜索
3. 做 `VARPRO` 风格的分解优化
4. 如果还不够好，再进入全参数 fallback：
   - TRF
   - CMA
   - DE
5. 最后用 `L-BFGS-B` 做局部精修
6. 对有理数/幂指数参数做 snap，再尝试 refit

因此 `Structure` 的角色不是“某个优化器”，而更像“优化调度器”。

### 5.4 Optimizer 在搜索中的真实位置

每个候选公式最终都是通过 `_eval_single()` 进入 `Evaluator.evaluate_skeleton()`（[`src/search.py:47-82`](../src/search.py#L47)）。

在 `_evaluate_all_parents()` 里，所有候选会被统一丢进线程池并行评估（[`src/search.py:693-881`](../src/search.py#L693)）。

评估一个候选的完整过程是：

1. 解析公式
2. 调用优化器拟合常数
3. 得到最佳训练误差
4. 用拟合后的参数回填表达式
5. 计算 `test/ood` 误差
6. 把结果更新回 `EvolutionTree`

这说明 Optimizer 决定的是“同一结构能被拟合到多好”，从而直接影响 Selector 下一轮看到的排序。

## 6. 一轮搜索中四个组件如何协同

以 `TreeSearch.run()`（[`src/search.py:363-623`](../src/search.py#L363)）为主线，一轮迭代可以拆成下面的时序：

### 步骤 1：Selector 选父节点

- 从 `EvolutionTree` 取摘要
- `selector.plan(...)` 选出若干 parent

### 步骤 2：Mutator 生成候选孩子

对每个 parent：

- `ASTMutator` 做删除/添加变异
- `LLMMutator` 生成补充型结构建议
- 统一规范化并去重

### 步骤 3：Optimizer 拟合每个候选

对每个 child：

- `Evaluator` 调用选定 `Optimizer`
- 产出 `train/test/ood` NMSE

### 步骤 4：Generator 补解释

对于那些比 parent 更优的 child：

- `Generator.describe_batch()` 补结构描述
- 描述写回节点，供后续 Selector 使用

### 步骤 5：Tree 更新状态

- 记录 child 分数
- 标记 `mature` 节点
- 标记 `degenerated` 节点
- 记录本轮选择历史

这 4 个核心组件形成的是一个闭环：

```text
Generator 给出起点和解释
-> Selector 决定扩展方向
-> Mutator 生成新结构
-> Optimizer/Evaluator 给结构打分
-> Generator 再给优秀结果补解释
-> 下一轮 Selector 继续决策
```

## 7. 为什么这四个组件是系统核心

如果从“搜索系统最小闭环”看，这四个组件刚好对应四种不可替代的能力：

- `Generator`：没有它，就没有高质量起点和可读上下文
- `Selector`：没有它，就只能盲目扩展，预算利用率会很差
- `Mutator`：没有它，就没有结构空间探索
- `Optimizer`：没有它，就无法判断某个结构本身到底好不好

再换句话说：

- `Generator` 决定搜索从哪里开始、节点如何被理解
- `Selector` 决定搜索资源投向哪里
- `Mutator` 决定搜索空间如何被展开
- `Optimizer` 决定每个结构能否被公平评估

`main.py` 的工作，就是把这四种能力拼成一个可执行的搜索闭环。

## 8. 阅读源码时的推荐顺序

如果你准备继续往下读代码，推荐顺序是：

1. [`main.py`](../main.py)
2. [`src/search.py`](../src/search.py)
3. [`src/generator.py`](../src/generator.py)
4. [`src/selector.py`](../src/selector.py)
5. [`src/mutator.py`](../src/mutator.py)
6. [`src/evaluator.py`](../src/evaluator.py)
7. [`src/optimizer/structure.py`](../src/optimizer/structure.py)
8. [`src/evolution_tree.py`](../src/evolution_tree.py)

这样阅读时会先看到装配，再看到调度，然后再下沉到每个组件内部。
