import { useCallback, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, queryKeys, type ChartPoint, type Market, type MarketEvent } from "../api";
import { BlockHeading, Notice, Segmented, changeColor, price, signed } from "./shared";

type Range = "1m" | "3m";
const RANGES: { key: Range; label: string; days: number }[] = [
  { key: "1m", label: "1M", days: 21 },
  { key: "3m", label: "3M", days: 66 },
];

const PAD = { left: 8, right: 52, top: 8, bottom: 18 };
const UP = "text-green-600 dark:text-green-500";

// Callback ref: the chart box mounts only after data loads, so a ref read in an effect would miss it.
function useWidth() {
  const [width, setWidth] = useState(0);
  const observer = useRef<ResizeObserver | null>(null);
  const ref = useCallback((node: HTMLDivElement | null) => {
    observer.current?.disconnect();
    if (!node) return;
    observer.current = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width));
    observer.current.observe(node);
  }, []);
  return [ref, width] as const;
}

interface Series {
  key: keyof ChartPoint;
  label: string;
  className: string;
  dashed?: boolean;
}

function niceTicks(min: number, max: number, count = 4): number[] {
  const step = (max - min) / count;
  return Array.from({ length: count + 1 }, (_, i) => min + step * i);
}

/** One panel, one y-scale; panels share the x-axis and hover index. */
function Panel({ points, width, height, series, domain, ticks, guides = [], bars, hover, onHover, format, showDates }: {
  points: ChartPoint[];
  width: number;
  height: number;
  series: Series[];
  domain: [number, number];
  ticks?: number[];
  guides?: number[];
  bars?: keyof ChartPoint;
  hover: number | null;
  onHover: (i: number | null) => void;
  format: (v: number) => string;
  showDates?: boolean;
}) {
  const plotW = Math.max(width - PAD.left - PAD.right, 10);
  const plotH = height - PAD.top - PAD.bottom;
  const [lo, hi] = domain;
  const x = (i: number) => PAD.left + (points.length < 2 ? plotW / 2 : (i / (points.length - 1)) * plotW);
  const y = (v: number) => PAD.top + (1 - (v - lo) / (hi - lo || 1)) * plotH;
  const path = (key: keyof ChartPoint) =>
    points
      .map((p, i) => (p[key] == null ? null : `${x(i)},${y(p[key] as number)}`))
      .reduce<string[]>((segs, pt, i, all) => {
        if (pt) segs.push(`${i === 0 || !all[i - 1] ? "M" : "L"}${pt}`);
        return segs;
      }, [])
      .join(" ");
  const barW = Math.max(plotW / points.length - 2, 1);
  const dateTicks = [0, Math.floor((points.length - 1) / 2), points.length - 1];

  const handleMove = (e: React.PointerEvent<SVGRectElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const ratio = (e.clientX - rect.left) / rect.width;
    onHover(Math.round(Math.min(Math.max(ratio, 0), 1) * (points.length - 1)));
  };

  return (
    <svg width={width} height={height} className="block overflow-visible">
      {(ticks ?? niceTicks(lo, hi, 3)).map((t) => (
        <g key={t}>
          <line x1={PAD.left} x2={PAD.left + plotW} y1={y(t)} y2={y(t)} className="stroke-border" strokeWidth={1} />
          <text x={PAD.left + plotW + 6} y={y(t)} dy="0.32em" className="fill-muted-foreground text-[10px] font-mono">
            {format(t)}
          </text>
        </g>
      ))}
      {guides.map((g) => (
        <line key={g} x1={PAD.left} x2={PAD.left + plotW} y1={y(g)} y2={y(g)}
          className="stroke-muted-foreground/60" strokeDasharray="3 3" strokeWidth={1} />
      ))}
      {bars && points.map((p, i) => {
        const v = p[bars] as number | null;
        if (v == null) return null;
        return (
          <rect key={p.date} x={x(i) - barW / 2} width={barW} y={Math.min(y(v), y(0))} height={Math.max(Math.abs(y(v) - y(0)), 1)}
            className={v >= 0 ? "fill-green-500/50" : "fill-red-500/50"} />
        );
      })}
      {series.map((s) => (
        <path key={s.key} d={path(s.key)} fill="none" strokeWidth={s.dashed ? 1.5 : 2}
          strokeDasharray={s.dashed ? "4 3" : undefined} strokeLinejoin="round" className={s.className} />
      ))}
      {showDates && dateTicks.map((i, n) => (
        <text key={i} x={x(i)} y={height - 4} textAnchor={n === 0 ? "start" : n === 2 ? "end" : "middle"}
          className="fill-muted-foreground text-[10px] font-mono">
          {points[i]?.date.slice(5)}
        </text>
      ))}
      {hover != null && points[hover] && (
        <>
          <line x1={x(hover)} x2={x(hover)} y1={PAD.top} y2={PAD.top + plotH} className="stroke-foreground/40" strokeWidth={1} />
          {series.map((s) => {
            const v = points[hover][s.key] as number | null;
            return v == null ? null : (
              <circle key={s.key} cx={x(hover)} cy={y(v)} r={3.5} className={`${s.className} fill-background`} strokeWidth={2} />
            );
          })}
        </>
      )}
      <rect x={PAD.left} y={0} width={plotW} height={height} fill="transparent"
        onPointerMove={handleMove} onPointerLeave={() => onHover(null)} />
    </svg>
  );
}

function domainOf(points: ChartPoint[], keys: (keyof ChartPoint)[], padRatio = 0.06): [number, number] {
  const values = points.flatMap((p) => keys.map((k) => p[k] as number | null)).filter((v): v is number => v != null);
  if (!values.length) return [0, 1];
  const lo = Math.min(...values), hi = Math.max(...values);
  const pad = (hi - lo || Math.abs(hi) || 1) * padRatio;
  return [lo - pad, hi + pad];
}

const PRICE_SERIES: Series[] = [
  { key: "close", label: "Close", className: "stroke-foreground" },
  { key: "sma20", label: "SMA 20", className: "stroke-chart-2", dashed: true },
  { key: "sma50", label: "SMA 50", className: "stroke-chart-1", dashed: true },
];
const MACD_SERIES: Series[] = [
  { key: "macd", label: "MACD", className: "stroke-foreground" },
  { key: "signal", label: "Signal", className: "stroke-chart-1", dashed: true },
];

function Legend({ series, extra }: { series: Series[]; extra?: React.ReactNode }) {
  return (
    <div className="flex items-center gap-3 text-[10px] font-mono text-muted-foreground">
      {series.map((s) => (
        <span key={s.key} className="inline-flex items-center gap-1">
          <svg width={14} height={4}><line x1={0} x2={14} y1={2} y2={2} strokeWidth={2}
            strokeDasharray={s.dashed ? "3 2" : undefined} className={s.className} /></svg>
          {s.label}
        </span>
      ))}
      {extra}
    </div>
  );
}

function Tooltip({ point }: { point: ChartPoint | undefined }) {
  if (!point) return <span className="text-muted-foreground">Hover a chart for daily values</span>;
  const f = (v: number | null, d = 2) => (v == null ? "—" : v.toFixed(d));
  return (
    <span className="tabular-nums">
      {point.date} · close {f(point.close)} · SMA20 {f(point.sma20)} · SMA50 {f(point.sma50)} · RSI {f(point.rsi, 0)} · MACD {f(point.macd)}
    </span>
  );
}

type Signal = "Bullish" | "Bearish" | "Neutral";
const SIGNAL_CLASS: Record<Signal, string> = { Bullish: UP, Bearish: "text-destructive", Neutral: "text-muted-foreground" };

function snapshotRows(m: Market, last: number | null): { label: string; value: string; signal: Signal; note: string }[] {
  const s = m.summary;
  const vs = (avg: number | null | undefined, name: string) => {
    if (last == null || avg == null) return null;
    const gap = (last / avg - 1) * 100;
    return { label: `vs ${name}`, value: `${price(avg)} (${signed(gap)})`, signal: (gap > 0 ? "Bullish" : "Bearish") as Signal,
      note: gap > 0 ? "Price above average" : "Price below average" };
  };
  const rsi = s.rsi;
  const rows = [
    rsi == null ? null : {
      label: "RSI 14", value: rsi.toFixed(0),
      signal: (rsi >= 70 ? "Bearish" : rsi <= 30 ? "Bullish" : rsi >= 50 ? "Bullish" : "Bearish") as Signal,
      note: rsi >= 70 ? "Overbought" : rsi <= 30 ? "Oversold" : rsi >= 50 ? "Positive momentum" : "Weak momentum",
    },
    s.macd == null || s.signal == null ? null : {
      label: "MACD", value: `${s.macd.toFixed(2)} / ${s.signal.toFixed(2)}`,
      signal: (s.macd > s.signal ? "Bullish" : "Bearish") as Signal,
      note: s.macd > s.signal ? "Above signal line" : "Below signal line",
    },
    vs(s.sma20, "SMA 20"),
    vs(s.sma50, "SMA 50"),
    vs(s.sma200, "SMA 200"),
    s.high_52w == null || s.low_52w == null || last == null ? null : {
      label: "52-week range", value: `${price(s.low_52w)} – ${price(s.high_52w)}`, signal: "Neutral" as Signal,
      note: `${Math.round(((last - s.low_52w) / (s.high_52w - s.low_52w || 1)) * 100)}% of range`,
    },
  ];
  return rows.filter((r): r is NonNullable<typeof r> => r != null);
}

function EventsTable({ events }: { events: MarketEvent[] }) {
  if (!events.length) return <p className="text-xs text-muted-foreground">No scheduled events found.</p>;
  return (
    <table className="w-full text-xs">
      <thead className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
        <tr className="border-b border-border">
          <th className="text-left font-medium py-1.5 px-3">Date</th>
          <th className="text-left font-medium py-1.5 px-3">When</th>
          <th className="text-left font-medium py-1.5 px-3">Event</th>
          <th className="text-left font-medium py-1.5 px-3">Detail</th>
        </tr>
      </thead>
      <tbody>
        {events.map((e) => (
          <tr key={`${e.date}-${e.label}`} className={`border-b border-border last:border-0 ${e.days < 0 ? "text-muted-foreground" : ""}`}>
            <td className="py-1.5 px-3 font-mono tabular-nums whitespace-nowrap">{e.date}</td>
            <td className={`py-1.5 px-3 font-mono whitespace-nowrap ${e.days >= 0 && e.days <= 14 ? "text-destructive" : ""}`}>
              {e.days === 0 ? "today" : e.days > 0 ? `in ${e.days}d` : `${-e.days}d ago`}
            </td>
            <td className="py-1.5 px-3 font-medium whitespace-nowrap">{e.label}</td>
            <td className="py-1.5 px-3">{e.detail || "—"}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function MarketPanel({ ticker }: { ticker: string }) {
  const [range, setRange] = useState<Range>("3m");
  const [hover, setHover] = useState<number | null>(null);
  const [ref, width] = useWidth();
  const { data, isLoading, error } = useQuery({
    queryKey: queryKeys.market(ticker),
    queryFn: () => api.market(ticker),
    staleTime: 5 * 60_000,
  });

  if (isLoading) return <div className="h-64 bg-muted animate-pulse" />;
  if (error || !data) return <Notice label="Chart">{(error as Error)?.message ?? "No market data"}</Notice>;

  const days = RANGES.find((r) => r.key === range)!.days;
  const points = data.points.slice(-days);
  const last = points.at(-1)?.close ?? null;
  const changes = [
    { label: "1W", value: data.summary.change_1w },
    { label: "1M", value: data.summary.change_1m },
    { label: "3M", value: data.summary.change_3m },
    { label: "1Y", value: data.summary.change_1y },
  ];
  const macdDomain = domainOf(points, ["macd", "signal", "hist"], 0.1);

  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <BlockHeading title="Price & technicals">
          <Segmented options={RANGES} value={range} onChange={setRange} />
        </BlockHeading>
        <div className="border border-border grid grid-cols-4 gap-px bg-border">
          {changes.map((c) => (
            <div key={c.label} className="bg-background px-3 py-2">
              <div className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground">{c.label} change</div>
              <div className={`text-sm font-mono tabular-nums ${changeColor(c.value ?? null)}`}>{signed(c.value ?? null)}</div>
            </div>
          ))}
        </div>
        <div ref={ref} className="border border-border px-3 py-2 space-y-2">
          <div className="text-[11px] font-mono min-h-4"><Tooltip point={hover != null ? points[hover] : undefined} /></div>
          {width > 0 && points.length > 1 && (
            <>
              <Legend series={PRICE_SERIES} />
              <Panel points={points} width={width} height={200} series={PRICE_SERIES}
                domain={domainOf(points, ["close", "sma20", "sma50"])} hover={hover} onHover={setHover}
                format={(v) => v.toFixed(v >= 100 ? 0 : 2)} />
              <Legend series={[{ key: "rsi", label: "RSI 14", className: "stroke-chart-2" }]}
                extra={<span>· dashed lines at 30 / 70</span>} />
              <Panel points={points} width={width} height={80}
                series={[{ key: "rsi", label: "RSI 14", className: "stroke-chart-2" }]}
                domain={[0, 100]} ticks={[0, 30, 70, 100]} guides={[30, 70]} hover={hover} onHover={setHover} format={(v) => v.toFixed(0)} />
              <Legend series={MACD_SERIES} extra={<span>· bars = histogram</span>} />
              <Panel points={points} width={width} height={90} series={MACD_SERIES} bars="hist"
                domain={macdDomain} guides={[0]} hover={hover} onHover={setHover} format={(v) => v.toFixed(1)} showDates />
            </>
          )}
        </div>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <div className="space-y-2">
          <BlockHeading title="Technical snapshot" />
          <table className="w-full text-xs border border-border">
            <tbody>
              {snapshotRows(data, last).map((r) => (
                <tr key={r.label} className="border-b border-border last:border-0">
                  <td className="py-1.5 px-3 text-muted-foreground whitespace-nowrap">{r.label}</td>
                  <td className="py-1.5 px-3 font-mono tabular-nums whitespace-nowrap">{r.value}</td>
                  <td className={`py-1.5 px-3 font-medium ${SIGNAL_CLASS[r.signal]}`}>{r.signal}</td>
                  <td className="py-1.5 px-3 text-muted-foreground">{r.note}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="space-y-2">
          <BlockHeading title="Key dates" />
          <div className="border border-border"><EventsTable events={data.events} /></div>
        </div>
      </div>
    </div>
  );
}
