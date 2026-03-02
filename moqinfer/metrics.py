"""Metrics collection and reporting for MoQ vs REST benchmarks."""

from dataclasses import dataclass, field


def _percentile(values: list[float], p: float) -> float:
    """Compute the p-th percentile of a list of values (linear interpolation)."""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = (len(sorted_v) - 1) * p / 100
    lower = int(idx)
    upper = min(lower + 1, len(sorted_v) - 1)
    weight = idx - lower
    return sorted_v[lower] * (1 - weight) + sorted_v[upper] * weight


@dataclass
class RequestMetrics:
    """Per-request timing metrics."""

    connect_ms: float = 0.0
    ttft_ms: float = 0.0
    completion_ms: float = 0.0
    tool_rounds: int = 0
    tool_rtt_ms: list[float] = field(default_factory=list)
    total_tokens: int = 0


@dataclass
class BenchmarkResult:
    """Aggregate benchmark results for a single transport."""

    transport: str
    num_backends: int
    users_per_backend: int
    requests: list[RequestMetrics]
    wall_time_ms: float

    @property
    def avg_ttft_ms(self) -> float:
        values = [r.ttft_ms for r in self.requests if r.ttft_ms > 0]
        return sum(values) / len(values) if values else 0.0

    @property
    def p50_ttft_ms(self) -> float:
        return _percentile(
            [r.ttft_ms for r in self.requests if r.ttft_ms > 0], 50
        )

    @property
    def p99_ttft_ms(self) -> float:
        return _percentile(
            [r.ttft_ms for r in self.requests if r.ttft_ms > 0], 99
        )

    @property
    def avg_completion_ms(self) -> float:
        values = [r.completion_ms for r in self.requests]
        return sum(values) / len(values) if values else 0.0

    @property
    def avg_tool_rtt_ms(self) -> float:
        all_rtts = [rtt for r in self.requests for rtt in r.tool_rtt_ms]
        return sum(all_rtts) / len(all_rtts) if all_rtts else 0.0

    @property
    def total_tokens(self) -> int:
        return sum(r.total_tokens for r in self.requests)

    @property
    def throughput_tok_s(self) -> float:
        if self.wall_time_ms <= 0:
            return 0.0
        return self.total_tokens / (self.wall_time_ms / 1000)


def print_result(result: BenchmarkResult):
    """Print benchmark results for a single transport."""
    n = len(result.requests)
    print(
        f"\n  {result.transport.upper()} Results "
        f"({result.num_backends} backend x {result.users_per_backend} users, "
        f"{n} requests):"
    )
    print(f"  {'─' * 56}")
    for i, r in enumerate(result.requests):
        parts = [
            f"TTFT={r.ttft_ms:.0f}ms",
            f"completion={r.completion_ms:.0f}ms",
            f"tokens={r.total_tokens}",
        ]
        if r.tool_rounds:
            parts.append(f"{r.tool_rounds} tool round(s)")
        if r.tool_rtt_ms:
            rtts = ", ".join(f"{t:.0f}" for t in r.tool_rtt_ms)
            parts.append(f"tool_rtt=[{rtts}]ms")
        print(f"  Request {i + 1}: {', '.join(parts)}")
    print(f"  {'─' * 56}")
    print(f"  Wall time:      {result.wall_time_ms:.0f}ms")
    print(
        f"  Avg TTFT:       {result.avg_ttft_ms:.0f}ms "
        f"(p50={result.p50_ttft_ms:.0f}, p99={result.p99_ttft_ms:.0f})"
    )
    print(f"  Avg completion: {result.avg_completion_ms:.0f}ms")
    if result.avg_tool_rtt_ms > 0:
        print(f"  Avg tool RTT:   {result.avg_tool_rtt_ms:.0f}ms")
    print(
        f"  Throughput:     {result.throughput_tok_s:.1f} tok/s "
        f"({result.total_tokens} tokens)"
    )


def print_comparison(moq: BenchmarkResult, rest: BenchmarkResult):
    """Print side-by-side comparison table with speedup ratios."""

    def _speedup(moq_val: float, rest_val: float) -> str:
        """For latency metrics: rest/moq (>1 means MoQ is faster)."""
        if moq_val <= 0:
            return "N/A"
        return f"{rest_val / moq_val:.2f}x"

    def _speedup_inv(moq_val: float, rest_val: float) -> str:
        """For throughput metrics: moq/rest (>1 means MoQ is better)."""
        if rest_val <= 0:
            return "N/A"
        return f"{moq_val / rest_val:.2f}x"

    rows = [
        (
            "Wall time (ms)",
            f"{moq.wall_time_ms:.0f}",
            f"{rest.wall_time_ms:.0f}",
            _speedup(moq.wall_time_ms, rest.wall_time_ms),
        ),
        (
            "Avg TTFT (ms)",
            f"{moq.avg_ttft_ms:.0f}",
            f"{rest.avg_ttft_ms:.0f}",
            _speedup(moq.avg_ttft_ms, rest.avg_ttft_ms),
        ),
        (
            "P50 TTFT (ms)",
            f"{moq.p50_ttft_ms:.0f}",
            f"{rest.p50_ttft_ms:.0f}",
            _speedup(moq.p50_ttft_ms, rest.p50_ttft_ms),
        ),
        (
            "P99 TTFT (ms)",
            f"{moq.p99_ttft_ms:.0f}",
            f"{rest.p99_ttft_ms:.0f}",
            _speedup(moq.p99_ttft_ms, rest.p99_ttft_ms),
        ),
        (
            "Avg completion (ms)",
            f"{moq.avg_completion_ms:.0f}",
            f"{rest.avg_completion_ms:.0f}",
            _speedup(moq.avg_completion_ms, rest.avg_completion_ms),
        ),
        (
            "Avg tool RTT (ms)",
            f"{moq.avg_tool_rtt_ms:.0f}",
            f"{rest.avg_tool_rtt_ms:.0f}",
            _speedup(moq.avg_tool_rtt_ms, rest.avg_tool_rtt_ms),
        ),
        (
            "Throughput (tok/s)",
            f"{moq.throughput_tok_s:.1f}",
            f"{rest.throughput_tok_s:.1f}",
            _speedup_inv(moq.throughput_tok_s, rest.throughput_tok_s),
        ),
        (
            "Total tokens",
            f"{moq.total_tokens}",
            f"{rest.total_tokens}",
            "",
        ),
    ]

    w = 66
    print(f"\n{'═' * w}")
    print(f"  MoQ vs REST Comparison  (speedup = REST/MoQ for latency)")
    print(f"{'═' * w}")
    print(f"  {'Metric':<22} {'MoQ':>10} {'REST':>10} {'Speedup':>10}")
    print(f"  {'─' * 22}─{'─' * 10}─{'─' * 10}─{'─' * 10}")
    for label, mv, rv, ratio in rows:
        print(f"  {label:<22} {mv:>10} {rv:>10} {ratio:>10}")
    print(f"{'═' * w}")
