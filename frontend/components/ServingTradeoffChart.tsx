"use client";

import { useMemo } from "react";
import {
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { BenchmarkCell } from "@/lib/api";
import { backendColor, backendLabel, batchLabel, formatMs, formatRps, logTicks } from "@/lib/serving";

interface Props {
  cells: BenchmarkCell[];
  sloP99Ms: number;
  recommended: BenchmarkCell | null;
}

interface PlotPoint extends BenchmarkCell {
  recommended: boolean;
}

const SURFACE = "#1a1a1f"; // forge-card
const GRID = "#2a2a30"; // forge-border
const MUTED = "#5c5c66"; // forge-muted
const SECONDARY = "#9898a0"; // forge-secondary
const TEXT = "#ececef"; // forge-text
const ACCENT = "#6b9fce"; // forge-accent

function PointShape(props: { cx?: number; cy?: number; fill?: string; payload?: PlotPoint }) {
  const { cx, cy, fill, payload } = props;
  if (cx == null || cy == null || payload == null) return null;
  const emphasized = payload.recommended;
  return (
    <g>
      {/* Transparent hit area: pointer only has to get near the point */}
      <circle cx={cx} cy={cy} r={12} fill="transparent" />
      {emphasized && <circle cx={cx} cy={cy} r={11} fill="none" stroke={ACCENT} strokeWidth={2} />}
      <circle cx={cx} cy={cy} r={emphasized ? 6 : 4.5} fill={fill} stroke={SURFACE} strokeWidth={2} />
      {emphasized && (
        <text x={cx + 16} y={cy - 12} fill={TEXT} fontSize={12} fontWeight={600}>
          Recommended
        </text>
      )}
    </g>
  );
}

function PointTooltip({ active, payload }: { active?: boolean; payload?: Array<{ payload: PlotPoint }> }) {
  if (!active || !payload?.length) return null;
  const point = payload[0].payload;
  return (
    <div className="bg-forge-card border border-forge-border rounded-lg p-3 text-xs shadow-xl min-w-[190px]">
      <div className="flex items-center gap-2 mb-2">
        <span className="inline-block w-3 h-0.5 rounded" style={{ backgroundColor: backendColor(point.backend) }} />
        <span className="text-forge-secondary">
          {backendLabel(point.backend)} · batching {batchLabel(point)}
        </span>
      </div>
      <p className="text-forge-text font-semibold font-mono">{formatMs(point.p99_ms)} <span className="text-forge-muted font-normal">p99</span></p>
      <p className="text-forge-text font-mono">{formatRps(point.throughput_rps)}</p>
      <p className="text-forge-muted mt-1.5">
        p50 {formatMs(point.p50_ms)} · {point.concurrency} concurrent
      </p>
      {point.max_batch_size > 1 && (
        <p className="text-forge-muted">{point.mean_batch_rows.toFixed(1)} rows per forward pass</p>
      )}
      {point.errors > 0 && <p className="text-forge-error mt-1">{point.errors} errors</p>}
    </div>
  );
}

export default function ServingTradeoffChart({ cells, sloP99Ms, recommended }: Props) {
  const { series, yTicks, yDomain } = useMemo(() => {
    const byBackend = new Map<string, PlotPoint[]>();
    for (const cell of cells) {
      if (cell.p99_ms == null) continue;
      const point = { ...cell, recommended: recommended != null && cell === recommended };
      byBackend.set(cell.backend, [...(byBackend.get(cell.backend) ?? []), point]);
    }
    Array.from(byBackend.values()).forEach((points) => points.sort((a, b) => a.concurrency - b.concurrency));

    const values = [...cells.map((c) => c.p99_ms ?? NaN), sloP99Ms].filter((v) => Number.isFinite(v) && v > 0);
    const low = Math.min(...values) * 0.7;
    const high = Math.max(...values) * 1.4;
    return { series: Array.from(byBackend.entries()), yTicks: logTicks(low, high), yDomain: [low, high] as [number, number] };
  }, [cells, sloP99Ms, recommended]);

  if (series.length === 0) {
    return <p className="text-forge-muted text-sm">No results for this batching policy yet.</p>;
  }

  return (
    <div className="w-full">
      <div className="h-[380px]">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 20, right: 40, bottom: 28, left: 12 }}>
            <CartesianGrid stroke={GRID} />
            <XAxis
              dataKey="throughput_rps"
              type="number"
              domain={[0, "auto"]}
              tick={{ fill: MUTED, fontSize: 12 }}
              tickFormatter={(v: number) => (v >= 1000 ? `${(v / 1000).toFixed(v % 1000 ? 1 : 0)}K` : `${v}`)}
              stroke={GRID}
              label={{ value: "Throughput (req/s)", position: "insideBottom", offset: -16, fill: MUTED, fontSize: 12 }}
            />
            <YAxis
              dataKey="p99_ms"
              type="number"
              scale="log"
              domain={yDomain}
              ticks={yTicks}
              allowDataOverflow
              tick={{ fill: MUTED, fontSize: 12 }}
              tickFormatter={(v: number) => `${v} ms`}
              stroke={GRID}
              width={64}
              label={{ value: "p99 latency", angle: -90, position: "insideLeft", offset: 0, fill: MUTED, fontSize: 12 }}
            />
            <ReferenceLine
              y={sloP99Ms}
              stroke={SECONDARY}
              strokeDasharray="6 4"
              label={{ value: `p99 SLO ${sloP99Ms} ms`, position: "insideTopRight", fill: SECONDARY, fontSize: 11 }}
            />
            <Tooltip content={<PointTooltip />} cursor={false} isAnimationActive={false} />
            {series.map(([backend, points]) => (
              <Scatter
                key={backend}
                name={backendLabel(backend)}
                data={points}
                fill={backendColor(backend)}
                line={{ stroke: backendColor(backend), strokeWidth: 2 }}
                lineJointType="linear"
                shape={<PointShape />}
                isAnimationActive={false}
              />
            ))}
          </ScatterChart>
        </ResponsiveContainer>
      </div>
      <div className="flex flex-wrap items-center justify-center gap-6 mt-1 text-xs text-forge-secondary">
        {series.map(([backend]) => (
          <div key={backend} className="flex items-center gap-2">
            <span className="inline-block w-4 h-0.5 rounded" style={{ backgroundColor: backendColor(backend) }} />
            <span>{backendLabel(backend)}</span>
          </div>
        ))}
        <span className="text-forge-muted">
          Each line steps through {Math.min(...cells.map((c) => c.concurrency))} →{" "}
          {Math.max(...cells.map((c) => c.concurrency))} concurrent requests
        </span>
      </div>
    </div>
  );
}
