# `Optimizer` 逻辑拆解：Structure 常数优化链与 VARPRO 三阶段

本文聚焦 `symregression` 里的常数参数优化器，重点解释默认的 `Structure` 优化链。

对应代码主要在：

- [`src/optimizer/structure.py`](../src/optimizer/structure.py)
- [`src/optimizer/base.py`](../src/optimizer/base.py)
- [`src/optimizer/__init__.py`](../src/optimizer/__init__.py)

其中默认注册项里，`Structure` 就是主优化器入口，见 [`src/optimizer/__init__.py:18-24`](../src/optimizer/__init__.py#L18)。

## 1. `Structure` 优化器在做什么

`StructureOptimizer.optimize()` 定义在 [`src/optimizer/structure.py:32`](../src/optimizer/structure.py#L32)。

它的目标不是搜索表达式结构，而是：

- 在结构 `skeleton` 已经固定的前提下
- 对里面的常数参数 `c0, c1, ...` 做数值拟合
- 返回当前结构下尽可能好的参数和误差

从整体上看，`Structure` 采用的是一条分层优化链：

1. 先分析表达式里的参数约束、幂指数风险、正值约束、指数边界等
2. 先做和幂函数相关的定向预搜索
3. 优先尝试 `VARPRO` 风格分解优化
4. 如果还不够好，再退回全参数 fallback：`TRF -> CMA -> DE`
5. 最后用 `L-BFGS-B` 做局部精修
6. 结尾对有理数/幂指数参数做 snap，并尝试 refit

时间预算也按阶段切分，见 [`src/optimizer/structure.py:86-89`](../src/optimizer/structure.py#L86)：

- 前 45% 时间给“幂预搜索 + VARPRO”
- 中间 45% 时间给 fallback：`TRF / CMA / DE`
- 最后 10% 时间给 `L-BFGS-B`

## 2. 优化前的表达式分析

在真正开始优化前，`Structure` 会先把表达式解析成 SymPy 对象，并做一轮静态分析，见 [`src/optimizer/structure.py:41-58`](../src/optimizer/structure.py#L41)。

### 2.1 负底数幂与有理数约束

`detect_rational_constrained()` 定义在 [`src/optimizer/base.py:134`](../src/optimizer/base.py#L134)。

它会检查：

- 哪些参数出现在幂指数位置
- 这些幂的底数是否依赖某个 feature
- 训练集里该 feature 是否出现负值

如果“负底数 + 连续指数参数”同时成立，这类参数就会被记到 `rational_idx`。原因是普通实数幂 `x**a` 在 `x < 0` 且 `a` 不是简单有理数时很容易产生复数或数值异常。

随后 `make_safe_expr()` 会把这些风险 `Pow` 改写成 `_RealPow`，见 [`src/optimizer/base.py:164`](../src/optimizer/base.py#L164)。`_RealPow` 的实现见 [`src/optimizer/base.py:25-35`](../src/optimizer/base.py#L25)，它会先把指数 snap 到以 `1/3` 为步长的有理网格上，再在实数域里稳定求值。

### 2.2 正值约束与“小范围参数”提示

`param_hints()` 定义在 [`src/optimizer/base.py:184`](../src/optimizer/base.py#L184)。

它会返回两类 hint：

- `positive_idx`：更适合限制为正值的参数
- `small_idx`：更适合放在较小范围内搜索的参数

大致规则是：

- `log(...)` 里的参数倾向正值
- `exp(...)` 中直接线性进入指数的参数通常应该在较小范围内搜索
- 三角函数参数也通常放到小范围里
- 幂指数参数通常属于 `small_idx`

这些 hint 最终会进入 `smart_x0()` 和 `make_bounds()`，分别控制初值采样和参数边界，见 [`src/optimizer/base.py:450`](../src/optimizer/base.py#L450) 和 [`src/optimizer/base.py:476`](../src/optimizer/base.py#L476)。

### 2.3 指数参数边界与 offset 参数边界

`compute_exp_param_bounds()` 定义在 [`src/optimizer/base.py:268`](../src/optimizer/base.py#L268)。

它会根据训练数据范围动态收紧下面几类参数的搜索区间：

- `exp(c_i * f(X))` 里的指数系数
- `exp(g(feat - c_i))` 里的中心偏移参数
- `(feat - c_i)**n` 这类幂函数 offset 参数

这样做是为了避免指数爆炸，也避免 offset 参数飘到完全脱离数据分布的区域。

### 2.4 Gaussian / offset 网格提示

`detect_gaussian_offsets()` 定义在 [`src/optimizer/base.py:389`](../src/optimizer/base.py#L389)。

它会识别形如：

- `exp(-(x-c)**2)`
- `exp(g(x-c))`
- `(x-c)**n`

这样的“中心位置参数”，然后基于对应 feature 的训练集范围生成一组 dense grid，供后面的预搜索使用。

## 3. 主流程总览

`StructureOptimizer.optimize()` 的主链可以直接对应到代码里的注释块：

- Phase 0: Compound-Pow pre-search，见 [`src/optimizer/structure.py:116`](../src/optimizer/structure.py#L116)
- Phase 1: `VARPRO`，见 [`src/optimizer/structure.py:137`](../src/optimizer/structure.py#L137)
- Phase 2: 全参数 fallback，见 [`src/optimizer/structure.py:158`](../src/optimizer/structure.py#L158)
- Phase 3: `L-BFGS-B` refinement，见 [`src/optimizer/structure.py:220`](../src/optimizer/structure.py#L220)
- Phase 4: rational snap，见 [`src/optimizer/structure.py:233`](../src/optimizer/structure.py#L233)
- Phase 5: Pow exponent snap + refit，见 [`src/optimizer/structure.py:237`](../src/optimizer/structure.py#L237)

可以先把它压缩成一句话：

```text
表达式分析
  -> 幂函数定向预搜索
  -> VARPRO 分解优化
  -> 不够好时退回全参数 TRF/CMA/DE
  -> L-BFGS-B 精修
  -> snap 幂指数 / 有理参数并局部 refit
```

## 4. Phase 0：和幂函数相关的预搜索

这一阶段的入口在 [`src/optimizer/structure.py:116-135`](../src/optimizer/structure.py#L116)。

它处理的是“compound-pow”问题，也就是：

- 某个参数出现在幂指数位置
- 同时幂底里还含有其他参数

检测逻辑在 `_detect_compound_pow_params()`，见 [`src/optimizer/structure.py:260`](../src/optimizer/structure.py#L260)。

例如 `(x + c1)**c2`、`(c0 + x*y)**c3` 这类结构，连续优化往往特别难搜。于是 `_try_pow_presearch()` 会先把这类幂指数固定到一组常见候选值上，再对剩余参数跑 `VARPRO`，见 [`src/optimizer/structure.py:283`](../src/optimizer/structure.py#L283)。

候选值定义在 [`src/optimizer/structure.py:252-257`](../src/optimizer/structure.py#L252)：

- `1`
- `2`
- `1/2`
- `3`
- `1/3`
- `-1`
- `3/2`
- `4`
- `5`
- `-1/2`
- `-2`
- `-3`

这一步的核心目的不是直接得到最终答案，而是先把“高风险的连续幂指数”离散化成几个结构上更可信的候选值，帮助后续优化找到更好的 basin。

## 5. `VARPRO` 是什么

`VARPRO` 的入口在 [`src/optimizer/structure.py:858`](../src/optimizer/structure.py#L858)。

它的核心思想是：

- 把参数拆成“线性参数”和“非线性参数”
- 外层只搜索非线性参数
- 线性参数在每次评估时直接用 OLS 解出来

### 5.1 如何区分线性参数和非线性参数

`_identify_nonlinear_params()` 定义在 [`src/optimizer/structure.py:833`](../src/optimizer/structure.py#L833)。

凡是出现在这些位置的参数，都会被视为非线性参数：

- `sin/cos/tan` 的内部
- `exp` 的内部
- `log` 的内部
- `Pow` 或 `_RealPow` 的底数或指数

剩下那些只作为线性乘子出现的参数，会被归入线性参数。

### 5.2 `VARPRO` 怎么把表达式拆开

`_try_varpro()` 里会先对表达式做 `expand`，然后逐项拆成：

- `param_part`
- `data_part`

见 [`src/optimizer/structure.py:889-917`](../src/optimizer/structure.py#L889)。

如果一项里的 `param_part` 只含线性参数，而 `data_part` 只依赖数据和非线性参数，那么这项就能被视作一个 basis function。

固定一组非线性参数后，代码会：

1. 计算所有 basis 的列向量，拼成设计矩阵 `A`
2. 用 `np.linalg.lstsq(A, y_adj)` 直接解线性系数
3. 由此得到当前非线性参数下的最优线性解

对应代码见 [`src/optimizer/structure.py:964-994`](../src/optimizer/structure.py#L964)。

因此 `VARPRO` 本质上是在把原来的全参数联合优化，降成“只优化非线性参数”的问题。

## 6. `VARPRO` 的三个阶段分别是什么

`VARPRO` 内部又分成三段，这是理解它的关键。

### 6.1 Phase 1：幂指数有理网格搜索

入口在 [`src/optimizer/structure.py:1068`](../src/optimizer/structure.py#L1068)。

这一阶段只在 `VARPRO` 的非线性参数里存在幂指数参数时触发。流程是：

1. 找出哪些非线性参数出现在幂指数位置，见 [`src/optimizer/structure.py:1062-1063`](../src/optimizer/structure.py#L1062)
2. 对每个这类参数，枚举一组常见整数/有理数幂，候选定义在 [`src/optimizer/structure.py:404-407`](../src/optimizer/structure.py#L404)
3. 每固定一个候选值，就对剩余非线性参数跑 `TRF`
4. 线性参数仍然每次都由 OLS 直接求解

具体实现是 `_pow_rational_grid()`，见 [`src/optimizer/structure.py:473`](../src/optimizer/structure.py#L473)。

这一步想解决的问题是：

- 幂指数参数很容易导致目标函数非常崎岖
- 很多真实公式的指数其实就落在 `1/2`、`1/3`、`2`、`3` 这种简单值上
- 与其一上来把它当连续变量乱搜，不如先检查“简单幂次是否已经足够解释数据”

所以 Phase 1 更像“结构化离散预搜索”。

### 6.2 Phase 1b：Gaussian offset 网格扫描

入口在 [`src/optimizer/structure.py:1091`](../src/optimizer/structure.py#L1091)。

这一阶段只有在下面两个条件同时成立时才触发：

- 检测到了 Gaussian / offset 参数
- 没有进入 Pow Grid，也就是 `not nl_pow_idx`

其流程是：

1. 识别 `(feat - c)` 型中心偏移参数
2. 用训练数据范围生成一个 dense grid
3. 把这个 offset 参数逐个固定在网格点上
4. 对其余非线性参数跑 `TRF`
5. 比较哪一个中心位置效果最好

相关 grid 来自 [`src/optimizer/base.py:389-447`](../src/optimizer/base.py#L389)，而实际扫描逻辑在 [`src/optimizer/structure.py:1093-1154`](../src/optimizer/structure.py#L1093)。

它要解决的是“中心位置参数不好搜”的问题。像高斯中心、峰值位置、平移量这种参数，直接做连续优化往往非常敏感，先按数据范围扫一轮通常更稳。

### 6.3 Phase 2：连续非线性搜索

入口在 [`src/optimizer/structure.py:1157`](../src/optimizer/structure.py#L1157)。

这一步才是把剩余非线性参数当连续变量来优化，顺序是：

1. `TRF`
2. `CMA-ES`
3. `DE`

对应代码分别在：

- [`src/optimizer/structure.py:1178-1189`](../src/optimizer/structure.py#L1178)
- [`src/optimizer/structure.py:1191-1209`](../src/optimizer/structure.py#L1191)
- [`src/optimizer/structure.py:1211-1227`](../src/optimizer/structure.py#L1211)

这里优化的目标不是“全参数原始 MSE”，而是 `ols_objective(nl_values)`：

- 先固定一组非线性参数
- 再用 OLS 解线性参数
- 最后看整体误差

所以这一步依然是 `VARPRO` 语境下的连续优化，而不是全参数联合优化。

### 6.4 `VARPRO` 三阶段之间的关系

可以把这三段理解成：

- Phase 1：先问“幂指数是不是简单整数/分数”
- Phase 1b：再问“中心偏移是不是落在数据范围里的某些位置”
- Phase 2：最后把剩余非线性参数作为连续变量系统搜索

前两段更偏“有结构的定向预搜索”，第三段才是标准连续优化。

## 7. 为什么 `VARPRO` 后还要做全参数 fallback

主流程里的 fallback 入口在 [`src/optimizer/structure.py:158`](../src/optimizer/structure.py#L158)。

触发条件是：

- `VARPRO` 没给出可用初值，或
- 当前最优误差还没低到 `_early_mse` 以下

这里的 `_early_mse = 1e-10 * var(y_train)`，见 [`src/optimizer/structure.py:60-64`](../src/optimizer/structure.py#L60)。

虽然 `VARPRO` 很强，但它依赖表达式能被很好地拆成“线性部分 + 非线性部分”。一旦表达式耦合太强、拆分不理想，或者 OLS basis 本身病态，`VARPRO` 可能不够稳。

所以 `Structure` 会在必要时退回全参数优化，顺序是：

- `TRF`，见 [`src/optimizer/structure.py:164-180`](../src/optimizer/structure.py#L164)
- `CMA`，见 [`src/optimizer/structure.py:182-200`](../src/optimizer/structure.py#L182)
- `DE`，见 [`src/optimizer/structure.py:202-218`](../src/optimizer/structure.py#L202)

注意这里和 `VARPRO` 内部的 `TRF/CMA/DE` 不一样：

- `VARPRO` 内部只优化非线性参数，线性参数由 OLS 解
- 这里的 fallback 是直接对全部参数做优化，不再分解

## 8. `L-BFGS-B` 为什么放在最后

主流程里的局部精修入口在 [`src/optimizer/structure.py:220`](../src/optimizer/structure.py#L220)。

它会从当前 `best_params` 出发跑一轮 `L-BFGS-B`，必要时还会从 `parent_params` 再 warm-start 一轮，见 [`src/optimizer/structure.py:224-231`](../src/optimizer/structure.py#L224)。

之所以放在最后，是因为 `L-BFGS-B` 更适合：

- 在已经接近一个好解时做连续抛光
- 在边界约束下进一步压低误差

但它不擅长：

- 从很差的起点出发跨越复杂局部极小值
- 处理特别崎岖的全局搜索问题

所以前面先用 `VARPRO` 和 fallback 找 basin，最后再用 `L-BFGS-B` 精修，是比较自然的分工。

## 9. 最后的 snap 和 refit 在做什么

这部分在主流程末尾，见 [`src/optimizer/structure.py:233-244`](../src/optimizer/structure.py#L233)。

### 9.1 Rational snap

第一步是对 `rational_idx` 参数做 `_snap_rational()`，也就是按 `1/3` 网格 snap，见 [`src/optimizer/base.py:21-22`](../src/optimizer/base.py#L21)。

这一步的主要目的不是“让表达式更好看”，而是：

- 保证负底数幂在实数域里的稳定性
- 让后续 `_RealPow` 的语义和数值参数一致

### 9.2 Pow exponent snap + refit

第二步是 `_detect_pow_exponents()` 和 `_snap_pow_and_refit()`，见：

- [`src/optimizer/structure.py:409`](../src/optimizer/structure.py#L409)
- [`src/optimizer/structure.py:423`](../src/optimizer/structure.py#L423)

它会对所有出现在幂指数位置的参数做下面的操作：

1. 找到与当前值接近的整数/分数候选
2. 把指数固定到某个候选值
3. 在“固定该指数”的条件下再跑一次 `L-BFGS-B`
4. 只有 refit 之后误差更好，才接受这个 snap

这一步很重要，因为它不是简单地把 `0.487` 四舍五入成 `0.5` 就结束，而是会重新调整其他参数，判断“更简洁的幂次是否真的值得”。

## 10. 一句话总结整条优化链

`Structure` 的策略可以概括为：

- 先用表达式分析把搜索空间缩小到合理范围
- 先对幂指数和 offset 这类高风险参数做定向预搜索
- 优先用 `VARPRO` 把线性参数剥离掉，只优化非线性部分
- 如果分解优化不够好，再退回全参数 `TRF/CMA/DE`
- 最后用 `L-BFGS-B` 精修，并把幂指数整理到更稳定、更可解释的整数/分数形式

如果只看 `VARPRO` 内部三阶段，可以压成一句更短的话：

```text
Phase 1: 先试简单幂指数
Phase 1b: 再试中心偏移位置
Phase 2: 最后做连续非线性搜索
```
