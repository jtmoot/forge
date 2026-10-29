/**
 * Serving page helpers: backend identity colors, batching labels, and the
 * client-side mirror of the API's SLO recommendation (recommend_config in
 * forge/api/services/serving_bench.py) so the SLO input re-ranks instantly.
 */

import type { BenchmarkCell } from "@/lib/api";

/**
 * Categorical slots 1-3 (dark steps), validated all-pairs against the
 * forge-card surface. Color follows the backend, never its rank: the native
 * backend is always slot 1 whatever framework it is.
 */
const SLOT_COLORS = ["#3987e5", "#d95926", "#199e70"];
const NATIVE_BACKENDS = new Set(["pytorch", "xgboost", "sklearn"]);

export function backendColor(backend: string): string {
  if (NATIVE_BACKENDS.has(backend)) return SLOT_COLORS[0];
  if (backend === "onnxruntime") return SLOT_COLORS[1];
  return SLOT_COLORS[2];
}

export function backendLabel(backend: string): string {
  const labels: Record<string, string> = {
    pytorch: "PyTorch",
    xgboost: "XGBoost",
    sklearn: "scikit-learn",
    onnxruntime: "ONNX Runtime",
    "onnxruntime-int8": "ONNX Runtime int8",
  };
  return labels[backend] ?? backend;
}

export function isNative(backend: string): boolean {
  return NATIVE_BACKENDS.has(backend);
}

export function batchKey(cell: { max_batch_size: number; max_wait_ms: number }): string {
  return `${cell.max_batch_size}/${cell.max_wait_ms}`;
}

export function batchLabel(cell: { max_batch_size: number; max_wait_ms: number }): string {
  return cell.max_batch_size <= 1 ? "Off" : `${cell.max_batch_size} rows · ${cell.max_wait_ms} ms`;
}

export interface Recommendation {
  best: BenchmarkCell | null;
  candidates: BenchmarkCell[];
  lowestP99: BenchmarkCell | null;
}

/** Highest-throughput error-free cell whose p99 meets the SLO (ties: lower p99). */
export function recommend(cells: BenchmarkCell[], sloP99Ms: number): Recommendation {
  const clean = cells.filter((c) => c.errors === 0 && c.p99_ms != null);
  const candidates = clean
    .filter((c) => (c.p99_ms as number) <= sloP99Ms)
    .sort((a, b) => b.throughput_rps - a.throughput_rps || (a.p99_ms as number) - (b.p99_ms as number));
  const lowestP99 = clean.reduce<BenchmarkCell | null>(
    (lowest, c) => (lowest == null || (c.p99_ms as number) < (lowest.p99_ms as number) ? c : lowest),
    null,
  );
  return { best: candidates[0] ?? null, candidates, lowestP99 };
}

/** Naive serving at the same concurrency: native backend, batching off. */
export function baselineFor(cells: BenchmarkCell[], cell: BenchmarkCell): BenchmarkCell | null {
  return (
    cells.find((c) => isNative(c.backend) && c.max_batch_size <= 1 && c.concurrency === cell.concurrency) ?? null
  );
}

/** Clean 1-2-5 ticks covering [min, max] for a log axis. */
export function logTicks(min: number, max: number): number[] {
  const ticks: number[] = [];
  for (let exp = Math.floor(Math.log10(min)); exp <= Math.ceil(Math.log10(max)); exp++) {
    for (const mantissa of [1, 2, 5]) {
      const value = mantissa * 10 ** exp;
      if (value >= min * 0.999 && value <= max * 1.001) ticks.push(value);
    }
  }
  return ticks;
}

export function formatMs(value: number | null): string {
  if (value == null) return "—";
  return value < 10 ? `${value.toFixed(2)} ms` : `${value.toFixed(1)} ms`;
}

export function formatRps(value: number): string {
  return value >= 1000 ? `${(value / 1000).toFixed(1)}K req/s` : `${Math.round(value)} req/s`;
}
