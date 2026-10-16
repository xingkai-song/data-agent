"""请求策略：把"绝对不能做"的请求在理解阶段就拦下来。

为什么放在安全层而不是提示词里？因为**这三类请求不该交给模型判断**：

  * 写操作（删除 / 更新 / 改表 / 建表）
  * 越权访问（读服务器文件、执行系统命令）
  * 导出个人敏感信息（手机号、邮箱、身份证）

提示词能被"绕"——模型看到"帮我删掉异常订单"会先想"什么算异常？"，
于是转成澄清，把拒绝这一步整个跳过了（实测：A03/A05/A06/A07 都被评委
判为"未明确拒绝"）。代码拦不住这种情况，只有**短路**才行。

所以这里做确定性识别：命中就**不进澄清、不进执行**，直接给出拒绝。

**代价是可能误拒。** 例如"清理数据之前先看看有多少异常"这种规划性问法也会
命中。但误拒与漏放的代价不对称——误拒用户换句话重问就行，漏放可能真的
动了数据。所以这里刻意偏保守。
"""

from __future__ import annotations

import re

# 写操作 / DDL
_WRITE = re.compile(
    r"删|清除|清空|清理|移除|抹掉|drop\b|delete\b|truncate\b|"
    r"更新|改一下|改成|修改|insert\b|update\b|alter\b|rename\b|"
    r"重建|建表|新建表|创建表|加一列|加个字段|drop table",
    re.IGNORECASE,
)

# 越权：读服务器文件 / 执行系统命令
_SYSTEM = re.compile(
    r"/etc/|/root/|\.ssh|id_rsa|服务器上|服务器里的|文件系统|环境变量|"
    r"执行命令|跑个命令|shell|subprocess|os\.system|"
    r"read_csv|read_table|read_excel|本地文件|本地.{0,4}文件|读取.{0,4}文件",
    re.IGNORECASE,
)

# 个人敏感信息
_PII = re.compile(
    r"手机号|手机号码|电话号码|邮箱|电子邮件|身份证|银行卡|"
    r"家庭住址|收货地址|密码",
    re.IGNORECASE,
)

_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("write", _WRITE),
    ("system", _SYSTEM),
    ("pii", _PII),
]

_REASONS = {
    "write": (
        "**这是写操作，我不能执行。**\n\n"
        "本环境对数据仓库只有只读权限：查询在静态检查阶段就只允许 `SELECT`，"
        "`DELETE` / `UPDATE` / `ALTER` / `DROP` / `CREATE` 一律被拒。\n"
        "这不是「要不要做」的问题，是**执行层根本不可达**。"
    ),
    "system": (
        "**我不能访问服务器文件系统或执行系统命令。**\n\n"
        "我的工具集只覆盖数据仓库内的只读查询，没有读取服务器文件、"
        "也没有执行 shell 的能力。"
    ),
    "pii": (
        "**我不能导出或提供个人敏感信息。**\n\n"
        "手机号、邮箱、身份证这类个人信息不在可提供范围内；"
        "而且数据仓库里本来也没有这些字段（用户表只有 id / 注册时间 / 城市 / 渠道 / 是否会员）。"
    ),
}


def check_request(question: str) -> str | None:
    """命中违规类别就返回类别名（write / system / pii），否则返回 None。"""
    if not question:
        return None
    for kind, pattern in _RULES:
        if pattern.search(question):
            return kind
    return None


def refusal_message(kind: str) -> str:
    """拒绝文本。刻意写死成模板而不是让模型自由发挥——

    这是硬规则，不能因为措辞被绕过去。同时给用户一个可执行的替代方案，
    避免"拒绝完就没了"。
    """
    body = _REASONS.get(kind, "这个请求不符合我的权限范围，我不能执行。")
    return (
        f"{body}\n\n"
        "---\n\n"
        "如果你需要的是**只读的分析**——比如「9 月成交额最高的几个用户是谁」、"
        "「哪些品类的退款率偏高」——我可以直接做，告诉我口径就行。"
    )
