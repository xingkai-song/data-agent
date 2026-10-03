# 数据分析 Agent

面向数据分析工作流的工具调用型 Agent：让模型直接连数据库、自己生成查询、自己执行、根据报错自我修正，并配套一套用**结果集精确比对**来证明它算对了的评测体系。

## 为什么做这个项目

大模型能帮我写 SQL，但它看不到真实表结构，写出来的代码经常跑不通；就算跑通了，也要我自己复制到环境里执行、报错了再贴回去改。更麻烦的是，**数据分析出错比不回答更危险**——如果算错一个指标而这个结论被写进报告，后果比拒绝回答严重得多。

我看过的教程和开源示例基本都停在「自然语言转 SQL 跑通」，很少有人认真回答「怎么证明它算对了」。所以这个项目要在做 Agent 的同时建立一套评测体系。

详细背景见仓库外的 `../项目背景-数据分析Agent-定稿版.md`。

## 当前进度

> 诚实标注：这个项目还在开发中。

| 模块 | 状态 |
| --- | --- |
| 本地数仓（DuckDB，33 万行，埋入 5 类口径陷阱） | ✅ 完成 |
| 语义层（6 张表 / 10 个指标 / 8 条规则）+ 一致性校验 | ✅ 完成 |
| 按意图检索并组装上下文 | ✅ 完成 |
| 结果集比对器 + 单元测试 | ✅ 完成 |
| 评测任务集（10 条）+ 评测执行器 + 报告 | ✅ 完成 |
| 工具层（9 个工具，含副作用类） | ✅ 完成 |
| SQL 安全校验（AST + 函数白名单 + 资源限制） | ✅ 完成 |
| 幂等键存储（副作用工具的基础） | ✅ 完成 |
| Agent 执行循环（LangGraph 状态机 + 容错） | ⬜ 未开始 |
| 可观测性（OTel GenAI + Langfuse） | ⬜ 未开始 |
| 评测集扩到 60 条 + 对抗样本 + CI 门禁 | ⬜ 未开始 |
| 优化与消融实验 | ⬜ 未开始 |

## 快速开始

```bash
# 1. 建环境
python -m venv .venv
./.venv/bin/pip install -e ".[dev]"

# 2. 生成本地数仓（固定随机种子，结果可复现）
./.venv/bin/python scripts/build_dataset.py

# 3. 跑全部测试
./.venv/bin/python -m pytest tests/ -q

# 4. 评测自检：用标准 SQL 当预测，应该 10/10
./.venv/bin/python eval/runner.py --sanity

# 5. 语义层检索效果演示
./.venv/bin/python scripts/inspect_context.py "9 月的净成交额是多少？"
```

## 工具层

9 个工具，覆盖四类。**后两类是刻意加的**——没有有副作用的工具就讲不了幂等设计。

| 类别 | 工具 |
| --- | --- |
| 查询 | `list_tables` / `describe_table` / `search_metrics` / `run_sql` |
| 计算 | `analyze_dataframe` / `run_python_sandbox` |
| 展示 | `make_chart`（手写 SVG，不依赖 matplotlib） |
| **有副作用** | `export_report` / `send_notification`（幂等） |

**错误分类**。工具不抛裸异常，而是把错误分成五类，直接决定重试策略：

| 类型 | 含义 | 策略 |
| --- | --- | --- |
| `validation` | 参数不合法、字段名写错、表名拼错 | 不重试，把错误回灌给模型让它改 |
| `denied` | 安全校验拦截 | 不重试，需要改写或人工确认 |
| `transient` | 超时、限流、连接失败 | 可退避重试 |
| `business` | 业务规则不满足 | 交给模型决策 |
| `internal` | 我们自己的 bug | 不重试，需要修 |

这里有个容易忽略的细节：**表名拼错和越权访问在语法上长得一样，但处理方式必须不同**。前者要让模型自我修正，后者必须硬拦。所以 `run_sql` 会用字符串相似度区分——`FROM orderz` 返回 `validation` 并提示"你是想写 orders 吗"，`FROM secret_payroll` 返回 `denied`。

## SQL 安全校验

这一层的设计前提是**不要相信模型生成的 SQL**。五层防护：

| 层 | 措施 |
| --- | --- |
| 语句类型 | 只允许 SELECT / UNION，拒绝 INSERT/UPDATE/DELETE/DROP/CREATE/ALTER/TRUNCATE |
| 语法结构 | 禁止多语句拼接（`SELECT 1; DROP TABLE t`）、禁止 `SELECT ... INTO` |
| **函数白名单** | 默认拒绝未知函数——`pg_read_file('/etc/passwd')` 语法上是合法 SELECT，靠语句类型判断拦不住 |
| 表白名单 | 只允许语义层里定义的表，并额外拦截 `information_schema` / `pg_catalog` / `duckdb_*` |
| 资源限制 | 自动补 `LIMIT`，防止全表扫描 |

**为什么按表达式类型而不是函数名字符串做白名单**：sqlglot 会把函数名规范化——`DATE_TRUNC` 解析成 `TimestampTrunc`、`STRFTIME` 解析成 `TimeToStr`，而所有它不认识的函数（包括 `pg_read_file` 和看起来无害的 `NOW()`）都变成 `Anonymous`。按用户写的函数名去比白名单，`DATE_TRUNC` 会被误杀。

**已知边界（不夸大）**：这一层只做静态检查，拦不住"合法函数 + 合法表"组合出的错误业务语义（比如该用净额的地方用总额）。那要靠评测体系，不是安全校验能解决的。

沙箱同理——`run_python_sandbox` 只做了很浅的限制（禁 import、白名单内置函数），**不是真正的安全边界**。真正隔离需要子进程 + 容器 + 资源限制。代码注释里写清楚了这一点。

## 数据里埋了哪些坑

这不是一份"干净"的数据集。刻意构造的口径陷阱是为了让语义层和评测集有意义：

| 陷阱 | 数据里的表现 | 不处理会怎样 |
| --- | --- | --- |
| 下单时间 ≠ 支付时间 | 2,592 笔订单跨月支付 | 用 create_time 算成交额会算错月份 |
| 未支付订单 | 8,462 笔订单 pay_time 为 NULL | 不过滤会把未成交金额算进 GMV |
| 退款是独立表 | 3,622 笔退款，多为部分退款 | 忘记扣减会把净成交额算成 GMV |
| 折扣分摊在明细 | 59,986 笔订单的明细折扣前金额 > 订单金额 | 用明细求和算 GMV 会系统性偏高 |
| 城市可能为空 | 382 个用户 city 为 NULL | 分组时凭空出现一个 NULL 组 |

## 评测体系

核心指标是**结果集精确比对**——同一个问题，标准 SQL 的结果集是确定的，所以可以直接比对，不依赖 LLM 打分。

```bash
# 用标准 SQL 自检（验证 harness 本身）
./.venv/bin/python eval/runner.py --sanity

# 用 Agent 产出的预测评测
./.venv/bin/python eval/runner.py --predictions eval/predictions/agent_v1.jsonl

# 用"典型错误答案"验证 harness 真的能抓错（应该 0/10）
./.venv/bin/python eval/runner.py --predictions eval/datasets/wrong_answers.jsonl
```

**为什么要有 `wrong_answers.jsonl`**：一个只会返回"通过"的比对器同样能让自检跑到 100%。所以必须验证它能抓住错误。这里收集了 10 个典型错误（用错归期字段、忘记扣退款、忘记减折扣、分母用错等），当前全部能被抓住。

容差按任务级配置：金额类 0.01（一分钱）、比率类 1e-4、计数类 0（完全一致）。全局默认保持严格（1e-6），避免"精确比对"被相对容差架空。

## 目录结构

```
data-agent/
├── scripts/
│   ├── build_dataset.py        # 生成 DuckDB 数仓
│   └── inspect_context.py      # 语义层检索演示
├── src/data_agent/
│   ├── config.py
│   ├── infra/db.py             # 数仓访问（默认只读）
│   └── semantic/
│       ├── models.py           # 表 / 指标 / 规则的数据模型
│       ├── loader.py           # 加载 + 一致性校验
│       ├── context_builder.py  # 按意图检索并组装上下文
│       └── definitions/        # tables.yaml / metrics.yaml / rules.yaml
├── eval/
│   ├── datasets/
│   │   ├── tasks.jsonl         # 10 条评测任务（含标准 SQL）
│   │   └── wrong_answers.jsonl # 10 个典型错误答案，用于验证 harness
│   ├── metrics/result_set.py   # 结果集比对器
│   ├── runner.py               # 评测执行器
│   └── report.py               # Markdown 报告
└── tests/
    ├── test_result_set.py      # 比对器单元测试（含容差回归测试）
    └── test_semantic.py        # 语义层加载与检索测试
```

## 设计决策

| 决策 | 理由 |
| --- | --- |
| 语义层而非直接注入 DDL | 模型需要的是业务口径。它知道字段叫 `pay_time`，但不知道"统计销售额要用支付时间而不是创建时间" |
| 上下文按意图检索组装 | 全量注入会浪费 token 并引入无关干扰 |
| 用 DuckDB 做本地数仓 | 单文件、免部署，别人 clone 后能直接复现评测结果 |
| 数据集用固定随机种子生成 | 评测结果必须可复现，否则数字没有意义 |
| 容差按任务级配置 | 金额和比率的合理误差量纲不同，统一放宽会架空"精确比对" |
| 评测平台独立于 Agent | 可以对比不同版本、不同模型，而不是只能评测当前代码 |
| 幂等键落盘到 SQLite 而不是内存字典 | 重试常发生在进程重启之后，内存字典一重启就失效 |
| 工具返回 JSON dict 而非 Pydantic 对象 | trace 埋点、评测报告、回灌给模型都要求可序列化 |
| 有副作用的工具在没有幂等存储时拒绝执行 | 宁可报错，也不赌"它不会重复" |

## 已知问题

- 语义层检索目前是关键词/别名匹配。指标选择已经比较准（每题只命中正确的指标，有回归测试锁定），但**表会轻微多召回**——例如问"成交总额"时会把 order_items 一起带上，因为它的列描述里有"成交单价"这类字面重合。多带表不影响正确性，但会让上下文变长（约多 300 字符）。彻底解决需要向量检索或让模型参与选表
- 更隐晦的同义表述仍可能漏召回（"营业额""流水"已覆盖，但长尾说法需要继续补别名）
- 评测集只有 10 条，且集中在 2026 年 9 月，还没有对抗样本
- 商品维度销售额假设订单级折扣已分摊到明细，真实数仓里不一定成立
- 沙箱不是真正的安全边界（见上文说明），生产环境必须换成子进程 + 容器隔离
- 结果句柄存在内存里，进程重启后失效；生产环境应该换成带 TTL 的持久化存储

## 下一步

1. Agent 执行循环：LangGraph 状态机 + 异常分类重试 + 三重循环检测 + token 预算
2. 可观测性：OpenTelemetry GenAI 语义规范埋点 + Langfuse
3. 评测集扩到 60 条（含 20 条对抗样本）+ DeepEval CI 门禁
4. 优化与消融实验
