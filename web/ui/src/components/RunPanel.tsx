import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Loader2, Play, RefreshCw, Square } from "lucide-react";
import { api, queryKeys, type Models, type Run, type Rating } from "../api";
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
  tokens,
} from "./shared";

// Rough per-stock wall time; the agents make ~15–25 sequential LLM calls.
const MINUTES_PER_STOCK = { full: 6, quick: 4 };

interface ModelChoice {
  provider: string;
  deep: string;
  quick: string;
}

const MODEL_KEY = "ta.modelChoice";

function loadChoice(): ModelChoice | null {
  try {
    return JSON.parse(localStorage.getItem(MODEL_KEY) ?? "null");
  } catch {
    return null;
  }
}

/** Keeps the saved choice only while it still exists in the catalog. */
function validChoice(models: Models, choice: ModelChoice | null): ModelChoice {
  const provider = models.providers.find((p) => p.key === choice?.provider && p.available);
  if (provider && choice) {
    const deep = provider.deep.some((o) => o.value === choice.deep) ? choice.deep : provider.deep[0].value;
    const quick = provider.quick.some((o) => o.value === choice.quick) ? choice.quick : provider.quick[0].value;
    return { provider: provider.key, deep, quick };
  }
  return models.default;
}

const selectClass =
  "h-8 px-2 text-xs border border-border bg-background focus:outline-none focus:ring-1 focus:ring-ring min-w-0";

function ModelPicker({ models, value, onChange }: { models: Models; value: ModelChoice; onChange: (v: ModelChoice) => void }) {
  const provider = models.providers.find((p) => p.key === value.provider);
  return (
    <div className="border border-border grid grid-cols-1 sm:grid-cols-3 gap-px bg-border">
      <label className="bg-background px-3 py-2 space-y-1">
        <div className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground">Provider</div>
        <select
          value={value.provider}
          onChange={(e) => {
            const next = models.providers.find((p) => p.key === e.target.value)!;
            onChange({ provider: next.key, deep: next.deep[0].value, quick: next.quick[0].value });
          }}
          className={`${selectClass} w-full`}
        >
          {models.providers.map((p) => (
            <option key={p.key} value={p.key} disabled={!p.available}>
              {p.label}{p.available ? "" : " — not set up"}
            </option>
          ))}
        </select>
      </label>
      <label className="bg-background px-3 py-2 space-y-1">
        <div className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground" title="Research manager and portfolio manager — the final calls">
          Deep model · final decisions
        </div>
        <select value={value.deep} onChange={(e) => onChange({ ...value, deep: e.target.value })} className={`${selectClass} w-full`}>
          {provider?.deep.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
        </select>
      </label>
      <label className="bg-background px-3 py-2 space-y-1">
        <div className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground" title="Analysts, researchers, trader, risk debaters — most of the calls">
          Quick model · most calls
        </div>
        <select value={value.quick} onChange={(e) => onChange({ ...value, quick: e.target.value })} className={`${selectClass} w-full`}>
          {provider?.quick.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
        </select>
      </label>
    </div>
  );
}

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

function totalTokens(run: Run) {
  return run.items.reduce(
    (sum, i) => ({ input: sum.input + (i.tokens?.input ?? 0), output: sum.output + (i.tokens?.output ?? 0) }),
    { input: 0, output: 0 },
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
          {finished}/{run.items.length} done · {tokens(totalTokens(run))} tok
        </span>
        <span className="text-[10px] text-muted-foreground font-mono hidden md:inline">
          {run.config.deep_think_llm}
          {run.config.quick_think_llm !== run.config.deep_think_llm && ` / ${run.config.quick_think_llm}`}
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
                  {item.cached ? "cached" : item.started_at && item.finished_at ? duration(Math.round(item.finished_at - item.started_at)) : ""}
                </td>
                <td
                  className="px-3 py-1.5 text-xs font-mono tabular-nums text-muted-foreground hidden md:table-cell"
                  title={item.tokens ? `${item.tokens.calls} LLM calls · ${item.tokens.input.toLocaleString()} in / ${item.tokens.output.toLocaleString()} out` : undefined}
                >
                  {item.tokens && (item.status === "running" || item.status === "done") ? `${tokens(item.tokens)} tok` : ""}
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
  const [force, setForce] = useState(false);
  const [reuseFundamentals, setReuseFundamentals] = useState(true);
  const [choice, setChoice] = useState<ModelChoice | null>(null);

  const models = useQuery({ queryKey: queryKeys.models, queryFn: api.models, staleTime: Infinity });
  useEffect(() => {
    if (models.data && !choice) setChoice(validChoice(models.data, loadChoice()));
  }, [models.data, choice]);
  const updateChoice = (next: ModelChoice) => {
    setChoice(next);
    try {
      localStorage.setItem(MODEL_KEY, JSON.stringify(next));
    } catch {
      /* private mode */
    }
  };

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
        provider: choice!.provider,
        deep_model: choice!.deep,
        quick_model: choice!.quick,
        force,
        reuse_fundamentals: reuseFundamentals,
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
  const lastDate = useMemo(() => {
    const map = new Map<string, string>();
    analyses.data?.analyses.forEach((a) => map.set(a.ticker, a.trade_date));
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
  const alreadyDone = [...selected].filter((t) => lastDate.get(t) === tradeDate).length;

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

        {models.data && choice && <ModelPicker models={models.data} value={choice} onChange={updateChoice} />}

        <div className="flex items-center gap-x-4 gap-y-1.5 flex-wrap text-xs text-muted-foreground">
          <label
            className="flex items-center gap-1.5"
            title="Skips the aggressive/conservative/neutral risk debate. Faster and cheaper, but the risk case is thinner."
          >
            <input type="checkbox" checked={quick} onChange={(e) => setQuick(e.target.checked)} />
            Quick (skip risk debate)
          </label>
          <label className="flex items-center gap-1.5" title="Fundamentals only change on earnings; reuse a report from the last 7 days instead of re-running that analyst.">
            <input type="checkbox" checked={reuseFundamentals} onChange={(e) => setReuseFundamentals(e.target.checked)} />
            Reuse fundamentals &lt; 7 days old
          </label>
          <label className="flex items-center gap-1.5" title="By default a stock already analysed for this date is skipped and its saved result reused.">
            <input type="checkbox" checked={force} onChange={(e) => setForce(e.target.checked)} />
            Force re-run (ignore saved results)
          </label>
        </div>

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
          <button
            className={btnPrimary}
            onClick={() => start.mutate()}
            disabled={selected.size === 0 || busy || start.isPending || !choice}
            title={busy ? "Wait for the current run to finish" : undefined}
          >
            <Play className="h-3 w-3" />
            Analyze {selected.size || ""} {selected.size === 1 ? "stock" : "stocks"}
          </button>
        </div>

        {selected.size > 0 && (
          <Notice label="Estimate">
            Up to {estimate} minutes, one stock at a time, on {choice?.quick}
            {choice && choice.deep !== choice.quick && ` (final calls on ${choice.deep})`}.
            {!force && alreadyDone > 0 && ` ${alreadyDone} already analysed for ${tradeDate} will be reused at no cost.`}
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
