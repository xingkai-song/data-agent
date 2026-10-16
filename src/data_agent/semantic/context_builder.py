"""按意图检索语义层，组装成精简的上下文。

为什么不做全量注入：把 6 张表 + 14 个指标 + 8 条规则全塞进 prompt，
上下文会涨到 5,682 字符。按意图检索的中位数约 1,600 字符（120 条问题的
实测分布 356 ~ 2,544），也就是**约 3.5 倍**；端到端 token 实测**高约 44%**
（12 题对照，96,574 → 138,808）。

**注意：没有证据表明全量注入会让模型更容易选错字段。** 这里早先写过"实测
更差"，但仓库里并无对应实验；一次 12 题的对照实验显示，全量注入的结果集
正确率反而略高（9~10/12 vs 7/12，其中一条差异是比对器假阴性造成的）。
所以这个取舍是**成本**，不是准确率：检索把 6 次探索压缩成 1 步，省的是
token 和步数；检索失败时模型仍可调工具把口径补回来。检索召回提高之后，
全量注入的相对收益会自然消失。

评分设计（踩过两次坑之后定下来的）：

  1. **别名命中是最强信号，且长别名优先。**
     "净成交额" 包含 "成交额"，所以问净成交额时两个指标都会被命中，
     必须让长别名胜出，并剔除被更长别名包含的短别名。

  2. **文本相似度用重叠 token 数，不做长度归一化。**
     一开始按 `overlap / sqrt(描述长度)` 归一化，结果是描述越短的指标
     得分越高——net_gmv 的描述比 gmv 短，于是问"成交额"时 net_gmv 反而排第一。
     归一化的方向错了：该惩罚的是"描述长所以碰巧命中多"，而不是"命中多"。

  3. **必选表不能被裁掉。** 选中 net_gmv 就必须带上它 join 的 refunds，
     否则模型写不出正确的关联。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from data_agent.semantic.models import Metric, Rule, SemanticLayer, Table
from data_agent.semantic.compiler import CompilerError, MetricCompiler

_ASCII_WORD = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")
_CJK = re.compile(r"[\u4e00-\u9fff]+")

# 别名命中的基础分：保证"命中别名"永远压过"只是描述里有几个词重合"
_ALIAS_BASE = 1.0
# 别名每多一个字加的分，用于让长别名胜出
_ALIAS_LEN_BONUS = 0.1
# 描述里每多一个重合 token 加的分
_TEXT_BONUS = 0.15
# 只靠描述重合进入候选的门槛（至少两个词重合才算相关）
_MIN_DESCRIPTION_OVERLAP = 2
# 隔字命中的基础分。**必须低于精确命中**，保证"精确 > 隔字"的优先级：
# "净成交额" 精确命中 net_gmv 时，不能让 gmv 靠隔字反超。
_GAPPED_BASE = 0.6
# 隔字命中允许插入的最大字数。"VIP占比" 要能命中 "VIP成交额占比"（插 3 个字）。
_GAPPED_MAX_GAP = 3

# 可选表的相关性权重。表名/表注释是强信号，字段是中等信号，
# 补充说明(notes)是弱信号——notes 里常常只是"顺带提到"别的表。
_TABLE_W_NAME = 0.5
_TABLE_W_COLUMN = 0.3
_TABLE_W_NOTES = 0.1
_TABLE_MIN_SCORE = 0.25


def _tokens(text: str) -> set[str]:
    """把中英文混合文本切成可比较的 token。

    中文按字符二元组切分（"上月成交额" -> 上月/月成/成交/交额）；
    英文标识符按单词切分，并额外拆出下划线子词
    （"is_vip" -> is_vip / is / vip），这样问 "VIP" 也能命中列名。
    """
    text = text.lower()
    out: set[str] = set()
    for word in _ASCII_WORD.findall(text):
        out.add(word)
        if "_" in word:
            out.update(part for part in word.split("_") if part)
    for run in _CJK.findall(text):
        if len(run) == 1:
            out.add(run)
        for i in range(len(run) - 1):
            out.add(run[i : i + 2])
    return out


def _overlap_count(query_tokens: set[str], *texts: str) -> int:
    merged: set[str] = set()
    for t in texts:
        if t:
            merged |= _tokens(t)
    return len(query_tokens & merged)


def _gapped_match(
    alias: str, text: str, max_gap: int = _GAPPED_MAX_GAP
) -> tuple[int, int] | None:
    """允许别名中间插少量字的匹配。

    `VIP占比` 应该能命中 `VIP成交额占比`（中间插了"成交额"），
    `退款额` 应该能命中 `退款总额`。纯子串匹配做不到，这是实测里最常见的
    一类漏召回（新留出集上 `N19 VIP成交额占比` 就因为不连续而失配）。

    只对长度 >= 3 的别名启用：太短的别名做隔字匹配会大量误伤
    （比如 2 个字的别名几乎能在任何句子里当子序列出现）。

    返回命中的**区间**（起，止），而不是布尔值——上层要靠区间判断
    "谁被谁包含"，光比别名字符串会判错。
    """
    a, t = alias.lower(), text.lower()
    if len(a) < 3:
        return None
    for start, ch in enumerate(t):
        if ch != a[0]:
            continue
        j, i, gap = 1, start + 1, 0
        while i < len(t) and j < len(a):
            if t[i] == a[j]:
                j += 1
            else:
                gap += 1
                if gap > max_gap:
                    break
            i += 1
        if j == len(a):
            return (start, i)
    return None


def _required_tables(metric: Metric, layer: SemanticLayer) -> list[str]:
    """指标依赖哪些表——由编译器沿关系图推导，指标定义里不再手写 join。"""
    return MetricCompiler(layer).tables_for(metric)


def _table_relevance(qt: set[str], table: Table) -> float:
    """可选表的相关性打分。"""
    s = _TABLE_W_NAME * _overlap_count(qt, table.name, table.description)
    s += _TABLE_W_COLUMN * _overlap_count(
        qt,
        " ".join(c.name for c in table.columns),
        " ".join(c.description for c in table.columns),
    )
    s += _TABLE_W_NOTES * _overlap_count(qt, " ".join(table.notes))
    return s


@dataclass
class MetricHit:
    metric: Metric
    score: float
    matched_alias: str | None
    # 命中别名在问题里的区间（起，止）。用于"被更长命中包含就让位"的判断。
    span: tuple[int, int] | None = None


@dataclass
class Selection:
    """一次检索的结果，保留命中原因便于调试和评测。"""

    tables: list[Table] = field(default_factory=list)
    metrics: list[Metric] = field(default_factory=list)
    rules: list[Rule] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)

    def table_names(self) -> list[str]:
        return [t.name for t in self.tables]

    def metric_names(self) -> list[str]:
        return [m.name for m in self.metrics]


def _score_metrics(layer: SemanticLayer, question: str, qt: set[str]) -> list[MetricHit]:
    q_lower = question.lower()
    hits: list[MetricHit] = []

    for m in layer.metrics:
        # 别名 / 指标名 / 显示名，取命中的最长者
        candidates = [*m.aliases, m.name, m.display_name]
        matched: str | None = None
        span: tuple[int, int] | None = None
        for c in candidates:
            if not c:
                continue
            idx = q_lower.find(c.lower())
            if idx >= 0 and (matched is None or len(c) > len(matched)):
                matched, span = c, (idx, idx + len(c))

        # 精确子串没命中时，退一步试"中间隔字"。两者分开打分，
        # 保证精确命中永远压过隔字命中。
        gapped = False
        if matched is None:
            for c in candidates:
                if not c:
                    continue
                sp = _gapped_match(c, question)
                if sp and (matched is None or len(c) > len(matched)):
                    matched, span, gapped = c, sp, True

        overlap = _overlap_count(qt, m.description, " ".join(m.aliases), m.display_name)
        if matched is None and overlap < _MIN_DESCRIPTION_OVERLAP:
            continue

        score = 0.0
        if matched is not None:
            base = _GAPPED_BASE if gapped else _ALIAS_BASE
            score += base + _ALIAS_LEN_BONUS * len(matched)
        score += _TEXT_BONUS * overlap
        hits.append(MetricHit(metric=m, score=score, matched_alias=matched, span=span))

    # 命中区间长的优先：剔除"命中区间被另一个命中区间包含"的指标。
    # 例："净成交额" 命中时，靠 "成交额" 命中的 gmv 应该让位；
    # "VIP成交额占比" 里，靠隔字命中 "VIP占比" 的 vip_gmv_share 区间更长，
    # 靠精确命中 "VIP成交额" 的 vip_gmv 反而该让位。
    # 注意：必须按**在问题里的区间**比，不能按别名字符串比——后者判不出
    # "VIP成交额" 和 "VIP占比" 谁更贴合原句。
    dropped: set[str] = set()
    for a in hits:
        if a.span is None:
            continue
        for b in hits:
            if a is b or b.span is None:
                continue
            if a.span != b.span and b.span[0] <= a.span[0] and a.span[1] <= b.span[1]:
                dropped.add(a.metric.name)
                break

    return [h for h in hits if h.metric.name not in dropped]


def _prune_relative(scored: list, *, min_ratio: float) -> list:
    """只保留与最高分同量级的结果，避免"雨露均沾"式召回。"""
    if not scored:
        return []
    top = scored[0][0]
    if top <= 0:
        return []
    return [item for item in scored if item[0] >= top * min_ratio]


def select(
    layer: SemanticLayer,
    question: str,
    *,
    max_tables: int = 4,
    max_metrics: int = 3,
    metric_ratio: float = 0.75,
) -> Selection:
    """根据自然语言问题挑出相关的表、指标和规则。"""
    qt = _tokens(question)
    selection = Selection()

    # ---- 1) 指标 ----
    hits = _score_metrics(layer, question, qt)
    hits.sort(key=lambda h: -h.score)
    hits = _prune_relative([(h.score, h) for h in hits], min_ratio=metric_ratio)
    selection.metrics = [h.metric for _, h in hits[:max_metrics]]
    selection.scores.update(
        {f"metric:{h.metric.name}": h.score for _, h in hits}
    )

    # ---- 2) 表 ----
    # 必选表：选中指标依赖的表（基础表 + join 用到的表），绝不能因为分数低被裁掉
    required: list[str] = []
    for m in selection.metrics:
        for name in _required_tables(m, layer):
            if name not in required:
                required.append(name)

    # 可选表：问题关键词额外命中的表
    optional: list[tuple[float, str]] = []
    for t in layer.tables:
        if t.name in required:
            continue
        s = _table_relevance(qt, t)
        if s >= _TABLE_MIN_SCORE:
            optional.append((s, t.name))
    optional.sort(key=lambda x: -x[0])
    optional = _prune_relative(optional, min_ratio=0.5)

    chosen = list(required)
    for _, name in optional:
        if len(chosen) >= max_tables:
            break
        if name not in chosen:
            chosen.append(name)

    selection.tables = [t for t in (layer.table(n) for n in chosen) if t]
    selection.scores.update({f"table_required:{n}": 1.0 for n in required})
    selection.scores.update({f"table_optional:{n}": s for s, n in optional})

    # ---- 3) 规则：全局规则始终带上，再叠加命中指标（含其依赖）的专属规则 ----
    # 必须沿依赖展开：命中 net_gmv 时要带上 scope=gmv 的 R002（"成交额取
    # orders.amount，别用明细表求和"），否则模型只在子查询里写错了金额口径，
    # 规则却不在上下文里。
    scopes: list[str] = []
    queue = [m.name for m in selection.metrics]
    while queue:
        name = queue.pop(0)
        if name in scopes:
            continue
        scopes.append(name)
        dep = layer.metric(name)
        if dep is not None:
            queue.extend(dep.referenced_metrics())
    selection.rules = layer.rules_for(*scopes)
    return selection


def render_context(
    selection: Selection,
    *,
    layer: SemanticLayer | None = None,
    include_rules: bool = True,
    emit_sql: bool | None = None,
) -> str:
    """把检索结果渲染成可直接放进 prompt 的文本。

    传了 layer 就把编译出来的标准 SQL 一起给模型——模型拿到的是
    "已经推导好 join 和归期过滤的模板"，而不是一串需要自己拼的片段。

    `emit_sql=False` 时退回只给口径片段，用于消融对照：验证"注入编译 SQL"
    这个变量本身带来多少差异。默认值可由环境变量 SEMANTIC_EMIT_SQL 控制。

    默认**关闭**注入，依据是实测：60 条评测下，注入版答案正确率 73.3%、
    不注入版 75.0%，平均 token 基本一致（8706 vs 8842）。这个变量没有
    可测量的收益，所以不作为默认行为。编译器本身的价值在架构层面
    （join 不手写、口径单一来源、编译产物可交叉校验），不在提示词体积。
    模型需要完整 SQL 时仍然可以调 search_metrics 主动取，属于按需而非默认注入。
    """
    if emit_sql is None:
        emit_sql = os.getenv("SEMANTIC_EMIT_SQL", "0") == "1"
    parts: list[str] = []
    compiler = MetricCompiler(layer) if layer is not None else None

    if selection.tables:
        parts.append("## 相关表结构\n")
        for t in selection.tables:
            header = f"### {t.name} — {t.description}"
            if t.grain and t.grain not in t.description:
                header += f"（{t.grain}）"
            parts.append(header)
            for c in t.columns:
                line = f"- {c.name} {c.type}：{c.description}"
                if c.notes:
                    line += f"  ⚠ {c.notes}"
                if c.values:
                    vals = " / ".join(f"{k}={v}" for k, v in c.values.items())
                    line += f"  取值：{vals}"
                parts.append(line)
            for n in t.notes:
                parts.append(f"- 【注意】{n}")
            parts.append("")

    if selection.metrics:
        parts.append("## 指标口径\n")
        for m in selection.metrics:
            parts.append(f"### {m.name}（{m.display_name}）：{m.description}")
            parts.append(f"- 归期字段：{m.time_table_name()}.{m.time_column}")
            if m.is_derived:
                parts.append(
                    f"- 派生指标，由 {m.expression} 组合而成（已统一到上面的归期口径）"
                )
            sql = None
            if compiler is not None and emit_sql:
                try:
                    sql = compiler.compile(m.name).sql
                except CompilerError as exc:
                    parts.append(f"- 口径编译失败：{exc}")
            if sql:
                parts.append(
                    "- 标准 SQL（语义层编译生成；:start / :end 换成具体时间，"
                    "需要分组时自行加 GROUP BY）："
                )
                for line in sql.splitlines():
                    parts.append(f"    {line}")
            else:
                parts.append(f"- 计算：{m.definition or m.expression}")
                if m.filters:
                    parts.append(f"- 必须的过滤：{'; '.join(m.filters)}")
            for cv in m.caveats:
                parts.append(f"- 【易错】{cv}")
            parts.append("")

    if include_rules and selection.rules:
        parts.append("## 必须遵守的规则\n")
        for r in selection.rules:
            parts.append(f"- [{r.id}][{r.severity}] {r.description}")
        parts.append("")

    return "\n".join(parts).strip()


def build_context(layer: SemanticLayer, question: str, **kwargs) -> tuple[Selection, str]:
    """便捷入口：检索 + 渲染。"""
    selection = select(layer, question, **kwargs)
    return selection, render_context(selection, layer=layer)
