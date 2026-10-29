"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import PageHeader from "@/components/PageHeader";
import ServingTradeoffChart from "@/components/ServingTradeoffChart";
import SummaryCard from "@/components/SummaryCard";
import {
  fetchBenchmark,
  fetchBenchmarks,
  fetchModels,
  fetchServedModel,
  startBenchmark,
  type BenchmarkCell,
  type BenchmarkResponse,
  type ServedModelResponse,
} from "@/lib/api";
import {
  backendColor,
  backendLabel,
  baselineFor,
  batchKey,
  batchLabel,
  formatMs,
  formatRps,
  isNative,
  recommend,
} from "@/lib/serving";

const DEFAULT_SLO_MS = 20;
const POLL_MS = 2000;
const SUBTITLE = "Online inference for production models: backends, micro-batching, and load tests";

function LineKey({ backend }: { backend: string }) {
  return <span className="inline-block w-3 h-0.5 rounded shrink-0" style={{ backgroundColor: backendColor(backend) }} />;
}

export default function ServingPage() {
  const [modelName, setModelName] = useState<string | null>(null);
  const [served, setServed] = useState<ServedModelResponse | null>(null);
  const [benchmark, setBenchmark] = useState<BenchmarkResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [runError, setRunError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const [sloMs, setSloMs] = useState(DEFAULT_SLO_MS);
  const [batchFilter, setBatchFilter] = useState<string | null>(null);
  const [showAll, setShowAll] = useState(false);

  // Resolve the production model, load it for serving, and fetch its latest benchmark
  useEffect(() => {
    fetchModels()
      .then(async ({ models }) => {
        const production = models.find((m) => m.production_version != null);
        if (!production) throw new Error("No model is in production yet");
        setModelName(production.name);
        const [model, list] = await Promise.all([
          fetchServedModel(production.name),
          fetchBenchmarks(production.name),
        ]);
        setServed(model);
        const latest = list.benchmarks.find((b) => b.status !== "failed");
        if (latest) setBenchmark(await fetchBenchmark(latest.id));
      })
      .catch((err) => setError(err instanceof Error ? err.message : "Failed to load"))
      .finally(() => setLoading(false));
  }, []);

  // Poll a running benchmark so the chart fills in cell by cell
  useEffect(() => {
    if (benchmark?.status !== "running") return;
    const timer = setTimeout(() => {
      fetchBenchmark(benchmark.id).then(setBenchmark).catch(() => undefined);
    }, POLL_MS);
    return () => clearTimeout(timer);
  }, [benchmark]);

  const runBenchmark = useCallback(() => {
    if (!modelName) return;
    setStarting(true);
    setRunError(null);
    startBenchmark(modelName)
      .then((started) => {
        setBenchmark(started);
        setBatchFilter(null);
      })
      .catch((err) => setRunError(err instanceof Error ? err.message : "Could not start benchmark"))
      .finally(() => setStarting(false));
  }, [modelName]);

  const cells = useMemo(() => benchmark?.results ?? [], [benchmark]);
  const recommendation = useMemo(() => recommend(cells, sloMs), [cells, sloMs]);
  const best = recommendation.best;
  const baseline = best ? baselineFor(cells, best) : null;
  const speedup = best && baseline && baseline.throughput_rps > 0 ? best.throughput_rps / baseline.throughput_rps : null;

  const batchOptions = useMemo(() => {
    const seen = new Map<string, BenchmarkCell>();
    for (const cell of cells) if (!seen.has(batchKey(cell))) seen.set(batchKey(cell), cell);
    return Array.from(seen.entries());
  }, [cells]);
  const activeBatch = batchFilter ?? (best ? batchKey(best) : batchOptions[0]?.[0] ?? null);
  const chartCells = cells.filter((c) => batchKey(c) === activeBatch);

  const totalCells = benchmark?.params
    ? (benchmark.params.backends?.length ?? served?.backends.length ?? 0) *
      benchmark.params.batch_configs.length *
      benchmark.params.concurrency.length
    : 0;
  const running = benchmark?.status === "running";

  const runButton = (
    <button
      onClick={runBenchmark}
      disabled={running || starting || !modelName}
      className="px-4 py-2 text-sm font-medium rounded-lg bg-forge-accent/15 text-forge-accent border border-forge-accent/30 hover:bg-forge-accent/25 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
    >
      {running ? "Load test running…" : starting ? "Starting…" : "Run load test"}
    </button>
  );

  if (loading) {
    return (
      <div>
        <PageHeader title="Serving" subtitle={SUBTITLE} />
        <div className="grid grid-cols-4 gap-5 mb-10">
          {Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="bg-forge-card border border-forge-border rounded-xl p-6 animate-pulse">
              <div className="h-3 w-24 bg-forge-bg-raised rounded mb-3" />
              <div className="h-7 w-16 bg-forge-bg-raised rounded" />
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (error || !served) {
    return (
      <div>
        <PageHeader title="Serving" subtitle={SUBTITLE} />
        <div className="bg-forge-error/10 border border-forge-error/20 rounded-lg px-4 py-3 text-sm text-forge-error">
          {error ?? "Model could not be loaded for serving"}
        </div>
      </div>
    );
  }

  const tableCells = showAll
    ? [...cells].sort((a, b) => b.throughput_rps - a.throughput_rps)
    : recommendation.candidates.slice(0, 8);

  return (
    <div>
      <PageHeader title="Serving" subtitle={SUBTITLE} actions={runButton} />

      {runError && (
        <div className="bg-forge-error/10 border border-forge-error/20 rounded-lg px-4 py-3 text-sm text-forge-error mb-6">{runError}</div>
      )}

      {/* KPI row */}
      <div className="grid grid-cols-4 gap-5 mb-8">
        <SummaryCard
          title="Production model"
          value={`v${served.version} · ${served.model_type.toUpperCase()}`}
          subtitle={`${served.model_name} · input ${served.input_shape.join("×")}`}
        />
        <SummaryCard
          title={`Best under ${sloMs} ms p99`}
          value={best ? formatRps(best.throughput_rps) : "—"}
          subtitle={
            best
              ? `${backendLabel(best.backend)} · batching ${batchLabel(best)} · p99 ${formatMs(best.p99_ms)}`
              : cells.length
                ? `No config meets the SLO; closest p99 ${formatMs(recommendation.lowestP99?.p99_ms ?? null)}`
                : "Run a load test to find out"
          }
        />
        <SummaryCard
          title="vs naive serving"
          value={speedup ? `${speedup.toFixed(1)}×` : "—"}
          subtitle={baseline ? `${backendLabel(baseline.backend)}, no batching, ${baseline.concurrency} concurrent` : "Throughput at the same concurrency"}
          accent={speedup != null && speedup > 1}
        />
        <SummaryCard
          title="Runtime"
          value={benchmark?.runtime?.cpu_limit ? `${benchmark.runtime.cpu_limit} vCPU` : "—"}
          subtitle={
            benchmark?.runtime
              ? `${benchmark.runtime.torch_threads} inference threads · ${benchmark.runtime.cpu_count} visible cores`
              : "Recorded with each load test"
          }
        />
      </div>

      {/* Backends with parity */}
      <div className="grid grid-cols-3 gap-5 mb-8">
        {served.backends.map((backend) => {
          const parity = served.parity.find((p) => p.backend === backend);
          const single = cells.find((c) => c.backend === backend && c.max_batch_size <= 1 && c.concurrency === 1);
          return (
            <div key={backend} className="bg-forge-card border border-forge-border rounded-xl p-5">
              <div className="flex items-center gap-2 mb-3">
                <LineKey backend={backend} />
                <p className="text-sm font-medium text-forge-text">{backendLabel(backend)}</p>
                <span className="ml-auto text-2xs uppercase tracking-widest text-forge-muted">
                  {isNative(backend) ? "native" : "parity passed"}
                </span>
              </div>
              <p className="text-2xl font-semibold font-mono text-forge-text">{formatMs(single?.p50_ms ?? null)}</p>
              <p className="text-xs text-forge-secondary mt-1">single-request p50</p>
              <p className="text-xs text-forge-muted mt-3 font-mono">
                {parity
                  ? `max |Δp| ${parity.max_abs_diff?.toExponential(1) ?? "—"} · labels ${((parity.label_agreement ?? 0) * 100).toFixed(1)}%`
                  : "reference outputs"}
              </p>
            </div>
          );
        })}
      </div>

      {/* Load test */}
      <div className="bg-forge-card border border-forge-border rounded-xl p-6 mb-8">
        <div className="flex items-start justify-between mb-5">
          <div>
            <h2 className="text-sm font-semibold text-forge-text">Latency vs throughput</h2>
            <p className="text-xs text-forge-secondary mt-1">
              p99 latency against throughput as load rises, per backend. Larger batches buy throughput with tail latency.
            </p>
          </div>
          {running && (
            <p className="text-xs text-forge-accent font-mono">
              {cells.length} / {totalCells || "?"} cells
            </p>
          )}
        </div>

        {cells.length === 0 ? (
          <div className="py-16 text-center">
            <p className="text-sm font-medium text-forge-text mb-1">
              {running ? "Warming up the first cell…" : "No load test yet"}
            </p>
            <p className="text-sm text-forge-secondary">
              A load test replays real test-split windows through every backend, batching policy and concurrency level.
            </p>
          </div>
        ) : (
          <>
            {/* Filters: one row above the chart */}
            <div className="flex flex-wrap items-center gap-2 mb-4">
              <span className="text-2xs uppercase tracking-widest text-forge-muted mr-1">Batching</span>
              {batchOptions.map(([key, cell]) => (
                <button
                  key={key}
                  onClick={() => setBatchFilter(key)}
                  className={`px-3 py-1 text-xs rounded-md border transition-colors ${
                    key === activeBatch
                      ? "bg-forge-accent/15 text-forge-accent border-forge-accent/30"
                      : "text-forge-secondary border-forge-border hover:text-forge-text"
                  }`}
                >
                  {batchLabel(cell)}
                </button>
              ))}
              <label className="ml-auto flex items-center gap-2 text-xs text-forge-secondary">
                p99 SLO
                <input
                  type="number"
                  min={1}
                  step={1}
                  value={sloMs}
                  onChange={(e) => setSloMs(Math.max(1, Number(e.target.value) || DEFAULT_SLO_MS))}
                  className="w-16 bg-forge-bg-raised border border-forge-border rounded-md px-2 py-1 text-forge-text font-mono text-xs"
                />
                ms
              </label>
            </div>
            <ServingTradeoffChart
              cells={chartCells}
              sloP99Ms={sloMs}
              recommended={best && batchKey(best) === activeBatch ? best : null}
            />
          </>
        )}
      </div>

      {/* Recommendation table (also the chart's table view) */}
      {cells.length > 0 && (
        <div className="bg-forge-card border border-forge-border rounded-xl overflow-hidden mb-6">
          <div className="flex items-center justify-between px-5 py-4 border-b border-forge-border">
            <h2 className="text-sm font-semibold text-forge-text">
              {showAll ? `All ${cells.length} configurations` : `Configurations meeting p99 ≤ ${sloMs} ms`}
            </h2>
            <button onClick={() => setShowAll((v) => !v)} className="text-xs text-forge-accent hover:underline">
              {showAll ? "Show SLO ranking" : `Show all ${cells.length}`}
            </button>
          </div>
          {tableCells.length === 0 ? (
            <p className="px-5 py-6 text-sm text-forge-secondary">
              Nothing meets {sloMs} ms. The lowest p99 measured was {formatMs(recommendation.lowestP99?.p99_ms ?? null)}.
            </p>
          ) : (
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-forge-border bg-forge-bg-raised">
                  {["Backend", "Batching", "Concurrency", "p50", "p99", "Throughput", "vs naive"].map((h) => (
                    <th key={h} className="px-5 py-3 text-left text-2xs font-medium text-forge-muted uppercase tracking-widest">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {tableCells.map((cell) => {
                  const base = baselineFor(cells, cell);
                  const isBest = cell === best;
                  return (
                    <tr
                      key={`${cell.backend}-${batchKey(cell)}-${cell.concurrency}`}
                      className={`border-b border-forge-border last:border-b-0 ${isBest ? "bg-forge-accent/[0.06]" : ""}`}
                    >
                      <td className="px-5 py-3 text-xs text-forge-text">
                        <span className="flex items-center gap-2">
                          <LineKey backend={cell.backend} />
                          {backendLabel(cell.backend)}
                          {isBest && <span className="text-2xs uppercase tracking-widest text-forge-accent ml-1">best</span>}
                        </span>
                      </td>
                      <td className="px-5 py-3 text-xs text-forge-secondary">{batchLabel(cell)}</td>
                      <td className="px-5 py-3 text-xs text-forge-secondary font-mono tabular-nums">{cell.concurrency}</td>
                      <td className="px-5 py-3 text-xs text-forge-secondary font-mono tabular-nums">{formatMs(cell.p50_ms)}</td>
                      <td className="px-5 py-3 text-xs text-forge-text font-mono tabular-nums">{formatMs(cell.p99_ms)}</td>
                      <td className="px-5 py-3 text-xs text-forge-text font-mono tabular-nums">{formatRps(cell.throughput_rps)}</td>
                      <td className="px-5 py-3 text-xs text-forge-secondary font-mono tabular-nums">
                        {base && base.throughput_rps > 0 ? `${(cell.throughput_rps / base.throughput_rps).toFixed(1)}×` : "—"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>
      )}

      {benchmark && (
        <p className="text-xs text-forge-muted">
          Load test {benchmark.id.slice(0, 8)} · {benchmark.status}
          {benchmark.completed_at && ` ${new Date(benchmark.completed_at).toLocaleString()}`} ·{" "}
          {benchmark.params?.requests_per_cell ?? "—"} requests per cell over{" "}
          {benchmark.params?.input_samples ?? "real"} test-split windows · latency is server-side (queue wait + forward
          pass; excludes network)
          {benchmark.error && <span className="text-forge-error"> · {benchmark.error}</span>}
        </p>
      )}
    </div>
  );
}
