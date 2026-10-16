"""Agent 执行循环（LangGraph 状态机）。

状态流转：

    understand ──(需要澄清)──> END（返回澄清问题）
        |
        +──> act ──> observe ──(发现问题)──> reflect ──> act
                        |
                        +──(完成)──> synthesize ──> END

几个刻意的设计选择：

  * **把"规划"折进 understand。** 原计划单独做一个 plan 节点，但为了产出
    三行计划多花一次模型调用不划算。现在 understand 的结构化输出里
    同时包含意图和计划，状态里仍保留 plan 字段。
  * **observe / reflect 不调模型。** 它们做的是确定性检查（结果是否为空、
    是否被截断），构造好提示回灌即可，不需要额外推理。
  * **synthesize 不调模型。** 结论文本在 act 阶段已经产出，synthesize
    只负责附加"数据来源"。这样既省一次调用，也保证溯源信息不被模型改写。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from langgraph.graph import END, StateGraph
from opentelemetry import trace as _otel_trace

from data_agent.agent.policies import Budget, LoopGuard, RetryPolicy
from data_agent.agent.prompts import (
    EMPTY_RESULT_HINT,
    SANITY_HINT,
    SYSTEM_PROMPT,
    TRUNCATED_HINT,
    UNDERSTAND_PROMPT,
    build_first_user_message,
)
from data_agent.agent.sanity import check_result
from data_agent.safety.request_policy import check_request, refusal_message
from data_agent.agent.state import AgentState, new_state
from data_agent.llm.base import LLMClient
from data_agent.observability.tracing import AgentTracer, setup_tracing
from data_agent.semantic.context_builder import build_context
from data_agent.tools.base import ErrorKind, ToolContext
from data_agent.tools.registry import ToolRegistry


@dataclass
class AgentRunResult:
    question: str
    status: str
    answer: str | None
    clarification: str | None
    stop_reason: str | None
    steps: int
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, float] = field(default_factory=dict)
    intent: dict[str, Any] = field(default_factory=dict)
    plan: list[str] = field(default_factory=list)
    selection: dict[str, Any] = field(default_factory=dict)
    elapsed_s: float = 0.0
    trace_id: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.status == "answered"


class DataAgent:
    def __init__(
        self,
        *,
        llm: LLMClient,
        registry: ToolRegistry,
        tool_ctx: ToolContext,
        loop_guard: LoopGuard | None = None,
        budget: Budget | None = None,
        retry: RetryPolicy | None = None,
        tracer: AgentTracer | None = None,
    ):
        self.llm = llm
        self.registry = registry
        self.tool_ctx = tool_ctx
        self.loop_guard = loop_guard or LoopGuard()
        self.budget = budget or Budget()
        self.retry = retry or RetryPolicy()
        # 默认关闭埋点，避免测试和一次性脚本产生额外开销
        self.tracer = tracer or setup_tracing(enabled=False)
        self._started_at = 0.0
        self._graph = self._build_graph()

    @property
    def _model_name(self) -> str:
        return getattr(self.llm, "model", "unknown")

    # ------------------------------------------------------------------
    # 图组装
    # ------------------------------------------------------------------
    def _build_graph(self):
        g = StateGraph(AgentState)
        g.add_node("understand", self._understand)
        g.add_node("act", self._act)
        g.add_node("observe", self._observe)
        g.add_node("reflect", self._reflect)
        g.add_node("synthesize", self._synthesize)

        g.set_entry_point("understand")
        g.add_conditional_edges(
            "understand",
            # 需要澄清、以及被安全策略拒绝时，都要走 synthesize——否则那些文字
            # 不会被组装成最终输出，用户什么都看不到（这个坑有回归测试守着）
            lambda s: (
                "synthesize"
                if (s.get("needs_clarification") or s.get("status") == "refused")
                else "act"
            ),
            {"synthesize": "synthesize", "act": "act"},
        )
        g.add_edge("act", "observe")
        g.add_conditional_edges(
            "observe",
            lambda s: s.get("_next", "act"),
            {"act": "act", "reflect": "reflect", "synthesize": "synthesize"},
        )
        g.add_edge("reflect", "act")
        g.add_edge("synthesize", END)
        return g.compile()

    # ------------------------------------------------------------------
    # 节点 1：意图理解（同时产出计划）
    # ------------------------------------------------------------------
    def _understand(self, state: AgentState) -> dict[str, Any]:
        question = state["question"]
        as_of = state.get("as_of") or time.strftime("%Y-%m-%d")

        # 安全策略短路：写操作 / 越权访问 / 敏感信息导出，在理解阶段就拦下，
        # 不进澄清、不进执行、不调模型。见 safety/request_policy.py
        violation = check_request(question)
        if violation:
            with self.tracer.node("understand") as span:
                span.set(**{"agent.policy_violation": violation})
            usage = dict(state.get("usage", {}))
            return {
                "intent": {"policy_violation": violation},
                "messages": [],
                "selection": {},
                "status": "refused",
                "stop_reason": f"请求被安全策略拦截：{violation}",
                "final_answer": refusal_message(violation),
                "usage": usage,
            }

        selection, context = build_context(self.tool_ctx.semantic, question)

        intent: dict[str, Any] = {}
        usage = dict(state.get("usage", {}))
        with self.tracer.node("understand") as span:
            span.set(**{"agent.metrics": selection.metric_names()})
            messages = [
                {"role": "system", "content": "你只输出 JSON。"},
                {
                    "role": "user",
                    "content": UNDERSTAND_PROMPT.format(question=question, as_of=as_of),
                },
            ]
            with self.tracer.llm_call(
                model=self._model_name, message_count=len(messages)
            ) as h:
                try:
                    resp = self.llm.complete(
                        messages, response_format={"type": "json_object"}
                    )
                    h.set_response(resp)
                    intent = _parse_json(resp.content)
                    usage = _add_usage(state, resp.usage, llm_calls=1)
                except Exception as exc:  # noqa: BLE001
                    # 意图识别失败不该让整个任务失败，降级为"直接执行"即可
                    h.fail(exc, kind=type(exc).__name__)
                    intent = {"error": f"{type(exc).__name__}: {exc}"}

        needs_clarification = bool(intent.get("needs_clarification"))
        plan = [str(x) for x in (intent.get("plan") or [])]
        first_user = build_first_user_message(question, context, plan)

        return {
            "intent": intent,
            "plan": plan,
            "needs_clarification": needs_clarification,
            "clarification": intent.get("clarification_question"),
            "context": context,
            "selection": {
                "tables": selection.table_names(),
                "metrics": selection.metric_names(),
                "rules": [r.id for r in selection.rules],
            },
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": first_user},
            ],
            "usage": usage,
            "status": "clarified" if needs_clarification else "running",
            "stop_reason": "需要澄清口径" if needs_clarification else None,
        }

    # ------------------------------------------------------------------
    # 节点 2：执行
    # ------------------------------------------------------------------
    def _act(self, state: AgentState) -> dict[str, Any]:
        stop = self.loop_guard.check(
            step=state.get("step", 0),
            tool_calls=state.get("tool_calls", []),
            observations=state.get("observations", []),
        ) or self.budget.exceeded(state.get("usage", {}), self._started_at)
        if stop:
            return {"status": "stopped", "stop_reason": stop, "_next": "synthesize"}

        # 用一个 step span 把"一次模型推理 + 它触发的工具调用"包在一起。
        # 这样回放时能直接看出"失败发生在第几步"，而不是在一堆平铺的
        # llm_call / execute_tool 里自己数。
        with self.tracer.node("step") as step_span:
            step_no = state.get("step", 0) + 1
            step_span.set(**{"agent.step": step_no})
            return self._act_step(state, step_no)

    def _act_step(self, state: AgentState, step_no: int) -> dict[str, Any]:
        messages = list(state.get("messages", []))
        schemas = self.registry.schemas()
        with self.tracer.llm_call(
            model=self._model_name,
            message_count=len(messages),
            tool_count=len(schemas),
        ) as h:
            try:
                resp = self.llm.complete(messages, schemas)
                h.set_response(resp)
            except Exception as exc:  # noqa: BLE001
                h.fail(exc, kind=type(exc).__name__)
                return {
                    "status": "failed",
                    "stop_reason": f"模型调用失败：{type(exc).__name__}: {exc}",
                    "errors": [
                        *state.get("errors", []),
                        {"stage": "act", "error": str(exc)},
                    ],
                    "_next": "synthesize",
                }

        usage = _add_usage(state, resp.usage, llm_calls=1)

        # 模型不再调用工具，认为它已经给出结论
        if not resp.wants_tools:
            return {
                "messages": messages
                + [{"role": "assistant", "content": resp.content or ""}],
                "final_answer": resp.content or "",
                "usage": usage,
                "status": "answered",
                "_next": "synthesize",
            }

        tool_calls = list(state.get("tool_calls", []))
        observations = list(state.get("observations", []))
        errors = list(state.get("errors", []))
        messages.append(
            {
                "role": "assistant",
                "content": resp.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                        },
                    }
                    for tc in resp.tool_calls
                ],
            }
        )

        executed = 0
        for tc in resp.tool_calls:
            executed += 1
            with self.tracer.tool_call(
                name=tc.name, call_id=tc.id, arguments=tc.arguments
            ) as th:
                result, attempts = self.retry.call(
                    lambda tc=tc: self.registry.execute(
                        tc.name, tc.arguments, self.tool_ctx
                    )
                )
                summary = _summarize(result)
                th.set(
                    **{
                        "agent.ok": result.ok,
                        "agent.attempts": attempts,
                        "agent.latency_ms": result.meta.get("latency_ms"),
                        "agent.row_count": summary.get("row_count"),
                        "agent.error_kind": (
                            result.error_kind.value if result.error_kind else None
                        ),
                    }
                )
                if not result.ok and result.error_kind is not None:
                    # 只有基础设施类故障才标记为 ERROR。
                    # 参数错误、安全拦截是"正常工作流的一部分"——
                    # 它们会被回灌给模型自我修正，不该污染失败率统计。
                    if result.error_kind in (ErrorKind.TRANSIENT, ErrorKind.INTERNAL):
                        th.fail(result.error or "工具失败", kind=result.error_kind.value)
                    else:
                        th.event(
                            "tool_error",
                            kind=result.error_kind.value,
                            message=(result.error or "")[:200],
                        )
            payload = result.model_dump(mode="json")
            # 结果体检：零成本的不变量检查，抓"跑通了但数明显不对"（见 sanity.py）
            data = payload.get("data")
            warnings = check_result(data)
            # 多路径对算的结论也一并回灌（工具在 query_tools 里算好，见 semantic/multipath.py）
            if isinstance(data, dict):
                warnings += [w for w in (data.get("cross_check") or []) if w not in warnings]
            tool_calls.append(
                {
                    "step": state.get("step", 0) + 1,
                    "name": tc.name,
                    "arguments": tc.arguments,
                    # 实际执行的 SQL：run_sql 在参数里，run_metric 在结果里。
                    # 评测和审计都靠这个字段，缺了就会漏掉编译路径的题。
                    "sql": _executed_sql(tc.name, tc.arguments, payload.get("data")),
                    "signature": LoopGuard.signature(tc.name, tc.arguments),
                    "ok": result.ok,
                    "error_kind": result.error_kind.value if result.error_kind else None,
                    "attempts": attempts,
                    "latency_ms": result.meta.get("latency_ms"),
                }
            )
            observations.append(
                {
                    "tool": tc.name,
                    "ok": result.ok,
                    "digest": LoopGuard.digest(payload.get("data")),
                    "summary": _summarize(result),
                    "warnings": warnings,
                }
            )
            if not result.ok:
                errors.append(
                    {
                        "stage": "tool",
                        "tool": tc.name,
                        "kind": result.error_kind.value if result.error_kind else None,
                        "error": result.error,
                    }
                )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": _tool_message(result),
                }
            )

        usage["tool_calls"] = usage.get("tool_calls", 0) + executed
        return {
            "messages": messages,
            "tool_calls": tool_calls,
            "observations": observations,
            "errors": errors,
            "usage": usage,
            "step": state.get("step", 0) + 1,
            "_next": "act",
        }

    # ------------------------------------------------------------------
    # 节点 3：结果校验（确定性检查）
    # ------------------------------------------------------------------
    def _observe(self, state: AgentState) -> dict[str, Any]:
        with self.tracer.node("observe") as span:
            if state.get("status") in ("answered", "failed", "stopped"):
                return {"_next": "synthesize"}
            if not state.get("tool_calls"):
                return {"_next": "act"}
            issue = _diagnose(state)
            span.set(**{"agent.diagnosis": issue})
            if issue:
                return {"diagnosis": issue, "_next": "reflect"}
            return {"_next": "act"}

    # ------------------------------------------------------------------
    # 节点 4：修正（构造回灌提示，不调模型）
    # ------------------------------------------------------------------
    def _reflect(self, state: AgentState) -> dict[str, Any]:
        hint = state.get("diagnosis")
        if not hint:
            return {}
        with self.tracer.node("reflect") as span:
            span.event("self_check", hint=hint[:200])
            messages = list(state.get("messages", []))
            messages.append({"role": "user", "content": f"【系统检查】{hint}"})
            reflections = list(state.get("reflections", []))
            reflections.append(hint)
            # 处理完要清空，否则下一次进入 observe 会读到旧的诊断结果
            return {"messages": messages, "reflections": reflections, "diagnosis": None}

    # ------------------------------------------------------------------
    # 节点 5：结论组装（附加数据来源）
    # ------------------------------------------------------------------
    def _synthesize(self, state: AgentState) -> dict[str, Any]:
        with self.tracer.node("synthesize") as span:
            status = state.get("status", "running")
            answer = state.get("final_answer")

            # refused 与 answered 走同一条：正文已经产生，直接用它。
            # 这样"拒绝"也是一次正常作答，而不是"未能完成分析"。
            if status in ("answered", "refused") and answer:
                body: str | None = answer
            elif status == "clarified":
                # 澄清分支必须把问题交回去。之前这里写成了 None，
                # 结果是"Agent 想澄清但用户什么都看不到"——
                # 功能上等同于没回答。
                body = state.get("clarification") or (
                    "这个问题存在多种口径，需要先确认你的口径后我才能给出准确结果。"
                )
            else:
                reason = state.get("stop_reason") or "未能完成分析"
                body = (
                    f"未能完成这次分析（原因：{reason}）。\n\n"
                    "已经执行的操作如下，可以据此继续排查：\n"
                    + "\n".join(
                        f"- {o.get('summary')}"
                        for o in state.get("observations", [])[-5:]
                    )
                )

            sources = ""
            if body:
                sources = _collect_sources(state)
                if sources:
                    body = f"{body}\n\n---\n\n**数据来源**\n\n{sources}"
            span.set(
                **{
                    "agent.status": status,
                    "agent.source_count": len(sources.splitlines()) if sources else 0,
                }
            )
            return {"final_answer": body}

    # ------------------------------------------------------------------
    # 对外入口
    # ------------------------------------------------------------------
    def run(self, question: str, *, as_of: str | None = None) -> AgentRunResult:
        self._started_at = time.time()
        started = time.perf_counter()
        with self.tracer.turn(question, as_of=as_of) as turn_span:
            final: dict[str, Any] = self._graph.invoke(
                new_state(question, as_of=as_of)
            )
            result = AgentRunResult(
                question=question,
                status=final.get("status", "unknown"),
                answer=final.get("final_answer"),
                clarification=final.get("clarification"),
                stop_reason=final.get("stop_reason"),
                steps=final.get("step", 0),
                tool_calls=final.get("tool_calls", []),
                observations=final.get("observations", []),
                usage=final.get("usage", {}),
                intent=final.get("intent", {}),
                plan=final.get("plan", []),
                selection=final.get("selection", {}),
                elapsed_s=time.perf_counter() - started,
            )
            # 记下 trace id：服务化之后要靠它把「这次请求」和「这次轨迹」关联起来
            span_ctx = _otel_trace.get_current_span().get_span_context()
            if span_ctx and span_ctx.is_valid:
                result.trace_id = f"{span_ctx.trace_id:032x}"
            turn_span.set(
                **{
                    "agent.status": result.status,
                    "agent.steps": result.steps,
                    "agent.tool_calls": len(result.tool_calls),
                    "agent.llm_calls": int(result.usage.get("llm_calls", 0)),
                    "agent.input_tokens": int(result.usage.get("input_tokens", 0)),
                    "agent.output_tokens": int(result.usage.get("output_tokens", 0)),
                    "agent.elapsed_s": round(result.elapsed_s, 3),
                    "agent.failed_tool_calls": sum(
                        1 for c in result.tool_calls if not c.get("ok")
                    ),
                }
            )
            if result.status != "answered":
                turn_span.event(
                    "not_answered",
                    status=result.status,
                    reason=(result.stop_reason or "")[:200],
                )
            return result


# ----------------------------------------------------------------------
# 辅助函数
# ----------------------------------------------------------------------


def _diagnose(state: AgentState) -> str | None:
    """对最后一次工具调用结果做确定性体检。"""
    observations = state.get("observations", [])
    if not observations:
        return None
    latest = observations[-1]
    if not latest.get("ok"):
        # 失败已经通过 tool 消息回灌，不重复处理
        return None
    summary = latest.get("summary") or {}
    if summary.get("empty") or summary.get("row_count") == 0:
        return EMPTY_RESULT_HINT
    if summary.get("truncated"):
        return TRUNCATED_HINT
    warnings = latest.get("warnings") or []
    if warnings:
        return SANITY_HINT.format(items="\n".join(f"- {w}" for w in warnings))
    return None


def _parse_json(text: str | None) -> dict[str, Any]:
    if not text:
        return {}
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[-1]
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def _add_usage(state: AgentState, usage: Any, *, llm_calls: int) -> dict[str, float]:
    cur: dict[str, float] = dict(state.get("usage", {}))
    cur["input_tokens"] = cur.get("input_tokens", 0) + getattr(usage, "input_tokens", 0)
    cur["output_tokens"] = cur.get("output_tokens", 0) + getattr(usage, "output_tokens", 0)
    cur["llm_calls"] = cur.get("llm_calls", 0) + llm_calls
    cur.setdefault("tool_calls", 0)
    return cur


def _summarize(result: Any) -> dict[str, Any]:
    """把工具结果压成一行摘要，用于轨迹记录和诊断。"""
    if not result.ok:
        return {
            "error": result.error,
            "kind": result.error_kind.value if result.error_kind else None,
        }
    data = result.data or {}
    if not isinstance(data, dict):
        return {"value": str(data)[:120]}
    summary: dict[str, Any] = {}
    for key in (
        "row_count",
        "empty",
        "handle",
        "truncated",
        "path",
        "replayed",
        "delivered",
        "rows_inlined",
        "more_rows_available",
    ):
        if key in data:
            summary[key] = data[key]
    return summary


def _executed_sql(name: str, args: Any, data: Any) -> str | None:
    """这次工具调用实际执行的 SQL。

    run_sql 的 SQL 在入参里；run_metric 的 SQL 是编译器产出的，在结果里
    （`compiled_sql`）。评测、审计、轨迹回放都依赖这个字段，所以统一暴露。
    """
    if isinstance(data, dict) and isinstance(data.get("compiled_sql"), str):
        return data["compiled_sql"]
    if isinstance(args, dict) and isinstance(args.get("sql"), str):
        return args["sql"]
    return None


def _tool_message(result: Any) -> str:
    if result.ok:
        return json.dumps(result.data, ensure_ascii=False, default=str)[:8000]
    parts = [f"错误类型：{result.error_kind.value}", f"错误信息：{result.error}"]
    if result.hint:
        parts.append(f"修正建议：{result.hint}")
    return "\n".join(parts)


def _collect_sources(state: AgentState) -> str:
    lines: list[str] = []
    for call in state.get("tool_calls", []):
        if call["name"] == "run_sql" and call.get("ok"):
            sql = str(call["arguments"].get("sql", "")).strip().replace("\n", " ")
            lines.append(f"- `{sql[:300]}`")
    return "\n".join(lines[:8])
