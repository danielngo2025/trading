import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Loader2, Play, RefreshCw, Square } from "lucide-react";
import { api, queryKeys, type Run, type Rating } from "../api";
import {
  BlockHeading,
  Notice,
  RatingBadge,
  Skeleton,
  StatusBadge,
  btnGhost,
  btnPrimary,
  changeColor,
  duration,
  marketCap,
  price,
  signed,
} from "./shared";

// Rough per-stock wall time; the agents make ~15–25 sequential LLM calls.
const MINUTES_PER_STOCK = { full: 6, quick: 4 };

function StageBar({ run, done, running }: { run: Run; done: number; running: boolean }) {
  return (
    <div className="flex gap-0.5">
      {run.stages.map((s, i) => (
        <div
          key={s.key}
          title={s.label}
          className={`h-1.5 w-5 ${
            i < done ? "bg-green-500" : running && i === done ? "bg-blue-500 animate-pulse" : "bg-muted"
          }`}
        />
      ))}
    </div>
  );
}

export function ActiveRun({ run, onSelect }: { run: Run; onSelect: (ticker: string) => void }) {
  const qc = useQueryClient();
  const cancel = useMutation({
    mutationFn: () => api.cancelRun(run.id),
    onSuccess: () => qc.invalidateQueries({ queryKey: queryKeys.runs }),
  });
  const finished = run.items.filter((i) => i.status === "done").length;
  const active = run.status === "queued" || run.status === "running";

  return (
    <div className="border border-border">
      <div className="flex items-center gap-2 px-3 py-2 bg-muted/50 flex-wrap">
        {active && <Loader2 className="h-3.5 w-3.5 animate-spin text-blue-500" />}
        <span className="text-sm font-medium">
          Run for {run.trade_date}
          {run.quick && <span className="text-[10px] font-mono text-muted-foreground ml-2">QUICK</span>}
        </span>
        <span className="text-xs text-muted-foreground font-mono tabular-nums">
          {finished}/{run.items.length} done
        </span>
        <div className="flex-1" />
        <StatusBadge status={run.status} />
        {active && (
          <button className={btnGhost} onClick={() => cancel.mutate()} disabled={run.cancel_requested || cancel.isPending}>
            <Square className="h-3 w-3" />
            {run.cancel_requested ? "Stopping after current stock" : "Stop"}
          </button>
        )}
      </div>
      {run.error && <div className="px-3 py-2 text-xs text-destructive">{run.error}</div>}
      <table className="w-full text-sm">
        <tbody>
          {run.items.map((item) => {
            const stage = run.stages.find((s) => s.key === item.stage);
            return (
              <tr
                key={item.ticker}
                className={`border-t border-border ${item.status === "done" ? "hover:bg-accent/30 cursor-pointer" : ""}`}
                onClick={() => item.status === "done" && onSelect(item.ticker)}
              >
                <td className="px-3 py-1.5 font-mono text-sm font-medium w-28">{item.ticker}</td>
                <td className="px-3 py-1.5">
                  <StageBar run={run} done={item.stages_done} running={item.status === "running"} />
                </td>
                <td className="px-3 py-1.5 text-[11px] text-muted-foreground hidden sm:table-cell">
                  {item.status === "running" ? stage?.label : item.status === "failed" ? (
                    <span className="text-destructive truncate block max-w-[320px]" title={item.error}>{item.error}</span>
                  ) : null}
                </td>
                <td className="px-3 py-1.5 text-xs font-mono tabular-nums text-muted-foreground hidden md:table-cell">
                  {item.started_at && item.finished_at ? duration(Math.round(item.finished_at - item.started_at)) : ""}
                </td>
                <td className="px-3 py-1.5 text-right">
                  {item.rating ? <RatingBadge rating={item.rating} /> : <StatusBadge status={item.status} />}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export function RunPanel({ activeRun, onSelect }: { activeRun: Run | null; onSelect: (ticker: string) => void }) {
  const qc = useQueryClient();
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [sector, setSector] = useState("all");
  const [tradeDate, setTradeDate] = useState(() => new Date().toISOString().slice(0, 10));
  const [quick, setQuick] = useState(false);

  const universe = useQuery({ queryKey: queryKeys.universe, queryFn: () => api.universe(), staleTime: Infinity });
  const analyses = useQuery({ queryKey: queryKeys.analyses, queryFn: api.analyses });
  const refresh = useMutation({
    mutationFn: () => api.universe(true),
    onSuccess: (data) => qc.setQueryData(queryKeys.universe, data),
  });
  const start = useMutation({
    mutationFn: () => {
      const stocks = universe.data?.stocks.filter((s) => selected.has(s.ticker)) ?? [];
      return api.createRun({
        tickers: stocks.map((s) => ({ ticker: s.ticker, company: s.company })),
        trade_date: tradeDate,
        quick,
      });
    },
    onSuccess: () => {
      setSelected(new Set());
      qc.invalidateQueries({ queryKey: queryKeys.runs });
    },
  });

  const lastRating = useMemo(() => {
    const map = new Map<string, Rating>();
    analyses.data?.analyses.forEach((a) => map.set(a.ticker, a.rating));
    return map;
  }, [analyses.data]);

  const stocks = universe.data?.stocks ?? [];
  const sectors = useMemo(() => [...new Set(stocks.map((s) => s.sector).filter(Boolean))].sort(), [stocks]);
  const visible = stocks.filter((s) => sector === "all" || s.sector === sector);
  const busy = activeRun != null && (activeRun.status === "queued" || activeRun.status === "running");

  const toggle = (ticker: string) =>
    setSelected((prev) => {
      const next = new Set(prev);
      next.has(ticker) ? next.delete(ticker) : next.add(ticker);
      return next;
    });

  const estimate = selected.size * MINUTES_PER_STOCK[quick ? "quick" : "full"];

  return (
    <div className="space-y-4">
      {activeRun && <ActiveRun run={activeRun} onSelect={onSelect} />}

      <div className="space-y-3">
        <BlockHeading title={universe.data ? `${universe.data.name} · by market cap` : "Universe"}>
          <button className={btnGhost} onClick={() => refresh.mutate()} disabled={refresh.isPending}>
            <RefreshCw className={`h-3 w-3 ${refresh.isPending ? "animate-spin" : ""}`} />
            Refresh prices
          </button>
        </BlockHeading>

        <div className="flex items-center gap-2 flex-wrap">
          <button
            className={btnGhost}
            onClick={() => setSelected(new Set(stocks.slice(0, 10).map((s) => s.ticker)))}
            disabled={stocks.length === 0}
          >
            Top 10 by cap
          </button>
          <button className={btnGhost} onClick={() => setSelected(new Set())} disabled={selected.size === 0}>
            Clear
          </button>
          <select
            value={sector}
            onChange={(e) => setSector(e.target.value)}
            className="h-8 px-2 text-xs border border-border bg-background focus:outline-none focus:ring-1 focus:ring-ring"
          >
            <option value="all">All sectors</option>
            {sectors.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
          <div className="flex-1" />
          <label className="flex items-center gap-1.5 text-xs text-muted-foreground">
            As of
            <input
              type="date"
              value={tradeDate}
              max={new Date().toISOString().slice(0, 10)}
              onChange={(e) => setTradeDate(e.target.value)}
              className="h-8 px-2 text-xs font-mono border border-border bg-background focus:outline-none focus:ring-1 focus:ring-ring"
            />
          </label>
          <label
            className="flex items-center gap-1.5 text-xs text-muted-foreground"
            title="Skips the aggressive/conservative/neutral risk debate. Faster and cheaper, but the risk case is thinner."
          >
            <input type="checkbox" checked={quick} onChange={(e) => setQuick(e.target.checked)} />
            Quick (skip risk debate)
          </label>
          <button
            className={btnPrimary}
            onClick={() => start.mutate()}
            disabled={selected.size === 0 || busy || start.isPending}
            title={busy ? "Wait for the current run to finish" : undefined}
          >
            <Play className="h-3 w-3" />
            Analyze {selected.size || ""} {selected.size === 1 ? "stock" : "stocks"}
          </button>
        </div>

        {selected.size > 0 && (
          <Notice label="Estimate">
            About {estimate} minutes, one stock at a time. Each stock runs the full analyst → debate → trader → risk
            pipeline on {activeRun?.config.deep_think_llm ?? "the configured model"}, so token cost scales with the
            count.
          </Notice>
        )}
        {start.error && <Notice label="Error">{(start.error as Error).message}</Notice>}

        {universe.isLoading ? (
          <Skeleton />
        ) : universe.error ? (
          <Notice label="Error">Could not load the TSX 60 list: {(universe.error as Error).message}</Notice>
        ) : (
          <div className="border border-border overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-accent/20 text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
                  <th className="px-3 py-2 w-8" />
                  <th className="px-3 py-2 text-left font-medium">#</th>
                  <th className="px-3 py-2 text-left font-medium">Ticker</th>
                  <th className="px-3 py-2 text-left font-medium hidden lg:table-cell">Sector</th>
                  <th className="px-3 py-2 text-left font-medium hidden sm:table-cell">Mkt cap</th>
                  <th className="px-3 py-2 text-left font-medium">Price</th>
                  <th className="px-3 py-2 text-left font-medium hidden sm:table-cell">Day</th>
                  <th className="px-3 py-2 text-left font-medium hidden md:table-cell">Last rating</th>
                </tr>
              </thead>
              <tbody>
                {visible.map((s) => (
                  <tr
                    key={s.ticker}
                    onClick={() => toggle(s.ticker)}
                    className={`border-t border-border cursor-pointer transition-colors ${
                      selected.has(s.ticker) ? "bg-accent/50" : "hover:bg-accent/30"
                    }`}
                  >
                    <td className="px-3 py-2">
                      <input type="checkbox" checked={selected.has(s.ticker)} readOnly className="pointer-events-none" />
                    </td>
                    <td className="px-3 py-2 text-xs font-mono tabular-nums text-muted-foreground">{s.rank}</td>
                    <td className="px-3 py-2">
                      <div className="font-mono text-sm font-medium">{s.ticker}</div>
                      <div className="text-[11px] text-muted-foreground truncate max-w-[200px]">{s.company}</div>
                    </td>
                    <td className="px-3 py-2 text-[11px] text-muted-foreground hidden lg:table-cell">{s.sector}</td>
                    <td className="px-3 py-2 text-xs font-mono tabular-nums hidden sm:table-cell">{marketCap(s.market_cap)}</td>
                    <td className="px-3 py-2 text-xs font-mono tabular-nums">{price(s.price, s.currency)}</td>
                    <td className={`px-3 py-2 text-xs font-mono tabular-nums hidden sm:table-cell ${changeColor(s.change_pct)}`}>
                      {signed(s.change_pct)}
                    </td>
                    <td className="px-3 py-2 hidden md:table-cell">
                      {lastRating.has(s.ticker) ? (
                        <RatingBadge rating={lastRating.get(s.ticker)!} />
                      ) : (
                        <span className="text-xs text-muted-foreground">—</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
