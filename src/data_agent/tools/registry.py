"""工具注册表。

用法：

    @tool(
        name="run_sql",
        description="执行只读 SQL 查询",
        category="query",
        input_model=RunSqlInput,
        output_model=RunSqlOutput,
    )
    def run_sql(inp: RunSqlInput, ctx: ToolContext) -> RunSqlOutput: ...

注册表负责三件事：把入参校验、执行、错误分类、审计日志统一起来，
让每个工具的实现只需要关心业务逻辑。
"""

from __future__ import annotations

import time
import traceback
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from data_agent.tools.base import (
    ErrorKind,
    ToolContext,
    ToolError,
    ToolResult,
    ToolSpec,
)


class ToolRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}

    # ---- 注册 ----
    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"工具重名：{spec.name}")
        self._specs[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        if name not in self._specs:
            raise KeyError(f"未注册的工具：{name}")
        return self._specs[name]

    def names(self) -> list[str]:
        return sorted(self._specs)

    def specs(self) -> list[ToolSpec]:
        return [self._specs[n] for n in self.names()]

    def schemas(self, *, categories: list[str] | None = None) -> list[dict[str, Any]]:
        """给模型看的工具清单。可以按类别裁剪，减少无关工具干扰。"""
        specs = self.specs()
        if categories:
            specs = [s for s in specs if s.category in categories]
        return [s.json_schema() for s in specs]

    def compact_schemas(
        self, *, categories: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """精简版工具清单。

        实测工具 schema 是每次 act 调用里最大的固定开销（约 1,639 token，
        比 system prompt 和语义层上下文加起来还大），而其中相当一部分是
        Pydantic 自动生成的噪声：

          * 每个字段都带一个 `title`（Sql / Purpose / Handle …），零信息量
          * 可选字段被包成 `anyOf: [{type: X}, {type: null}]`
          * 可选字段的 `default: null`

        这里只做**无损压缩**（不删任何有语义的约束）：去掉 title、
        把 anyOf-null 合并成 `type: [X, "null"]`、去掉空默认值。
        enum、description、required、min/max 这些一律保留——
        它们才是模型判断怎么调工具的依据。
        """
        out = []
        for schema in self.schemas(categories=categories):
            out.append({**schema, "parameters": _compact(schema["parameters"])})
        return out

    # ---- 执行 ----
    def execute(
        self,
        name: str,
        args: dict[str, Any] | BaseModel | None,
        ctx: ToolContext,
    ) -> ToolResult:
        try:
            spec = self.get(name)
        except KeyError as exc:
            # 模型选了一个不存在的工具，属于参数问题，回灌让它改
            return ToolResult.failure(
                ErrorKind.VALIDATION,
                str(exc),
                hint=f"可用工具：{', '.join(self.names())}",
            )

        started = time.perf_counter()

        # 1) 入参校验
        try:
            if isinstance(args, BaseModel):
                payload = args
            else:
                payload = spec.input_model(**(args or {}))
        except ValidationError as exc:
            return ToolResult.failure(
                ErrorKind.VALIDATION,
                f"参数校验失败：{exc.error_count()} 处错误",
                hint=_format_validation_error(exc, spec),
                errors=exc.errors(include_url=False),
            )
        except TypeError as exc:
            return ToolResult.failure(
                ErrorKind.VALIDATION,
                f"参数结构不合法：{exc}",
                hint=f"期望的参数结构：{spec.input_model.model_json_schema()}",
            )

        # 2) 执行
        try:
            out = spec.func(payload, ctx)
            if isinstance(out, ToolResult):
                result = out
            elif isinstance(out, BaseModel):
                # 统一转成 JSON 可序列化的 dict：
                # 后面要做 trace 埋点、写评测报告、回灌给模型，都要求可序列化。
                result = ToolResult.success(out.model_dump(mode="json"))
            else:
                result = ToolResult.success(out)
        except ToolError as exc:
            result = ToolResult.failure(exc.kind, str(exc), hint=exc.hint)
        except TimeoutError as exc:
            result = ToolResult.failure(
                ErrorKind.TRANSIENT, f"执行超时：{exc}", hint="可以稍后重试"
            )
        except (ConnectionError, OSError) as exc:
            result = ToolResult.failure(
                ErrorKind.TRANSIENT, f"外部依赖异常：{exc}", hint="可以稍后重试"
            )
        except Exception as exc:  # noqa: BLE001 - 兜底，避免工具异常打穿上层
            result = ToolResult.failure(
                ErrorKind.INTERNAL,
                f"{type(exc).__name__}: {exc}",
                hint="这是工具内部错误，不要重试同样的调用",
                traceback=traceback.format_exc()[-800:],
            )

        elapsed_ms = (time.perf_counter() - started) * 1000
        result.meta.setdefault("tool", name)
        result.meta.setdefault("category", spec.category)
        result.meta.setdefault("side_effect", spec.side_effect)
        result.meta.setdefault("latency_ms", round(elapsed_ms, 2))

        # 3) 审计
        if ctx.audit is not None:
            ctx.audit.record(
                tool=name,
                args=payload.model_dump(mode="json"),
                ok=result.ok,
                error_kind=result.error_kind.value if result.error_kind else None,
                error=result.error,
                latency_ms=elapsed_ms,
            )
        return result


def _format_validation_error(exc: ValidationError, spec: ToolSpec) -> str:
    """把校验错误转成模型能看懂、能据此修正的提示。"""
    lines = [f"调用 `{spec.name}` 的参数有误，请修正后重试："]
    for err in exc.errors(include_url=False):
        loc = ".".join(str(p) for p in err.get("loc", ()))
        lines.append(f"  - 字段 `{loc}`：{err.get('msg')}")
    lines.append(f"正确的参数结构：{spec.input_model.model_json_schema()}")
    return "\n".join(lines)


# Pydantic 自动生成、但对模型没有信息量的键
_NOISE_KEYS = {"title"}


def _compact(node: Any) -> Any:
    """递归精简 JSON Schema。只删噪声，不删约束。"""
    if isinstance(node, list):
        return [_compact(x) for x in node]
    if not isinstance(node, dict):
        return node

    out: dict[str, Any] = {}
    for key, value in node.items():
        if key in _NOISE_KEYS:
            continue
        # 可选字段的空默认值没有信息量
        if key == "default" and value is None:
            continue
        out[key] = _compact(value)

    # anyOf: [{type: T}, {type: null}] -> type: [T, "null"]
    any_of = out.get("anyOf")
    if isinstance(any_of, list) and len(any_of) == 2:
        null_part = next((x for x in any_of if x == {"type": "null"}), None)
        other = next((x for x in any_of if x != {"type": "null"}), None)
        if null_part is not None and isinstance(other, dict):
            merged = {k: v for k, v in other.items() if k != "type"}
            base_type = other.get("type")
            if isinstance(base_type, str):
                merged["type"] = [base_type, "null"]
            elif isinstance(base_type, list):
                merged["type"] = [*base_type, "null"]
            else:
                merged["type"] = "null"
            out.pop("anyOf")
            out = {**merged, **{k: v for k, v in out.items() if k != "type"}}
            out.setdefault("type", merged["type"])
    return out


# 全局默认注册表
DEFAULT_REGISTRY = ToolRegistry()


def tool(
    *,
    name: str,
    description: str,
    category: str,
    input_model: type[BaseModel],
    output_model: type[BaseModel],
    side_effect: bool = False,
    permission: str = "read",
    timeout_s: int = 30,
    registry: ToolRegistry | None = None,
) -> Callable:
    """把一个函数注册成工具。"""

    def deco(func: Callable) -> Callable:
        spec = ToolSpec(
            name=name,
            description=description,
            category=category,
            input_model=input_model,
            output_model=output_model,
            func=func,
            side_effect=side_effect,
            permission=permission,
            timeout_s=timeout_s,
        )
        (registry or DEFAULT_REGISTRY).register(spec)
        func.__tool_spec__ = spec  # type: ignore[attr-defined]
        return func

    return deco
