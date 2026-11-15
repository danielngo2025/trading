import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowRight, BellRing, Play, RefreshCw, X } from "lucide-react";
import { api, queryKeys, type AnalysisSummary, type Watch, type WatchEvent, type WatchLevel } from "../api";
import { BlockHeading, EmptyState, Notice, RatingBadge, Skeleton, btn, changeColor, price, signed } from "./shared";

const EVENT_LABEL: Record<WatchEvent["status"], string> = {
  triggered: "Waiting for you to re-run",
  waiting: "Queued to re-run",
  running: "Re-running…",
  done: "Re-rated",
  failed: "Re-run failed",
  skipped: "Not re-run (daily limit)",
};
const RERUNNABLE: WatchEvent["status"][] = ["triggered", "skipped", "failed"];

function when(ts: number): string {
  return new Date(ts * 1000).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

export function useWatch(refetchInterval = 30_000) {
  return useQuery({ queryKey: queryKeys.watch, queryFn: api.watch, refetchInterval });
}

function useRerun() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: api.watchRerun,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: queryKeys.watch });
      setTimeout(() => qc.invalidateQueries({ queryKey: queryKeys.runs }), 2000);
    },
  });
}

const input = "h-7 w-20 border border-border bg-background px-2 text-xs font-mono tabular-nums";

function NumberField({ value, onCommit, placeholder, step = "any" }: {
  value: number | null;
  onCommit: (v: number | null) => void;
  placeholder?: string;
  step?: string;
}) {
  const [draft, setDraft] = useState(value?.toString() ?? "");
  useEffect(() => setDraft(value?.toString() ?? ""), [value]);
  const commit = () => {
    const next = draft.trim() === "" ? null : Number(draft);
    if (next !== value && (next === null || Number.isFinite(next))) onCommit(next);
  };
  return (
    <input
      type="number" step={step} min="0" placeholder={placeholder} value={draft} className={input}
      onChange={(e) => setDraft(e.target.value)} onBlur={commit}
      onKeyDown={(e) => e.key === "Enter" && (e.target as HTMLInputElement).blur()}
    />
  );
}

/** "Re-run below / above" inputs for one stock, plus levels the decision itself names. */
function LevelEditor({ ticker, watch }: { ticker: string; watch: Watch }) {
  const qc = useQueryClient();
  const save = useMutation({
    mutationFn: (level: WatchLevel) => api.watchLevels(ticker, level),
    onSuccess: (data) => qc.setQueryData(queryKeys.watch, data),
  });
  const level = watch.levels[ticker] ?? { above: null, below: null };
  const set = (patch: Partial<WatchLevel>) => save.mutate({ ...level, ...patch });
  const suggestions = watch.suggestions[ticker] ?? [];
  return (
    <div className="space-y-1.5">
      <div className="flex items-center gap-x-4 gap-y-1.5 flex-wrap text-xs">
        <label className="flex items-center gap-1.5">
          Alert below <NumberField value={level.below} placeholder="—" onCommit={(v) => set({ below: v })} />
        </label>
        <label className="flex items-center gap-1.5">
          Alert above <NumberField value={level.above} placeholder="—" onCommit={(v) => set({ above: v })} />
        </label>
      </div>
      {suggestions.length > 0 && (
        <div className="flex items-center gap-1.5 flex-wrap text-[11px]">
          <span className="text-muted-foreground">From the decision:</span>
          {suggestions.map((s) => {
            const active = level[s.side] === s.price;
            return (
              <button
                key={`${s.side}${s.price}`}
                onClick={() => set({ [s.side]: active ? null : s.price })}
                className={`px-1.5 py-0.5 border font-mono ${active ? "border-primary bg-primary text-primary-foreground" : "border-border hover:bg-accent"}`}
                title={`Alert when the price goes ${s.side} ${s.price}`}
              >
                {s.label} {s.side === "below" ? "↓" : "↑"} {s.price}
              </button>
            );
          })}
        </div>
      )}
      {save.error && <p className="text-xs text-destructive">{(save.error as Error).message}</p>}
    </div>
  );
}

function RerunButton({ e }: { e: WatchEvent }) {
  const rerun = useRerun();
  if (!RERUNNABLE.includes(e.status)) return null;
  return (
    <button
      className={btn}
      disabled={rerun.isPending}
      onClick={(ev) => { ev.stopPropagation(); rerun.mutate(e.id); }}
    >
      <Play className="h-3 w-3" /> Re-run now
    </button>
  );
}

function EventRow({ e, onSelect, showTicker = true }: {
  e: WatchEvent;
  onSelect: (ticker: string) => void;
  showTicker?: boolean;
}) {
  const changed = e.new_rating && e.new_rating !== e.prev_rating;
  return (
    <div
      role="button"
      onClick={() => onSelect(e.ticker)}
      className={`flex items-center gap-3 px-3 py-2 border-t border-border first:border-t-0 hover:bg-accent/30 cursor-pointer flex-wrap ${
        e.seen ? "" : "bg-blue-50 dark:bg-blue-950/30"
      }`}
    >
      <span className="text-xs font-mono tabular-nums whitespace-nowrap shrink-0">{when(e.at)}</span>
      {showTicker && <span className="font-mono text-sm font-medium w-20">{e.ticker}</span>}
      <span className="text-xs text-muted-foreground flex-1 min-w-[180px]">
        {e.reason} · {price(e.base_price)} → {price(e.price)}
      </span>
      <span className="inline-flex items-center gap-1.5">
        <RatingBadge rating={e.prev_rating} />
        {e.new_rating && <><ArrowRight className="h-3 w-3" /><RatingBadge rating={e.new_rating} /></>}
      </span>
      <span className={`text-[11px] w-40 text-right ${changed ? "font-medium" : "text-muted-foreground"} ${
        e.status === "failed" ? "text-destructive" : ""}`} title={e.error}>
        {changed ? "Recommendation changed" : EVENT_LABEL[e.status]}
      </span>
      <RerunButton e={e} />
    </div>
  );
}

function headline(e: WatchEvent): string {
  if (e.status === "done") {
    return e.new_rating !== e.prev_rating
      ? `${e.ticker}: ${e.prev_rating} → ${e.new_rating}`
      : `${e.ticker}: re-rated, still ${e.new_rating}`;
  }
  return `${e.ticker} ${e.reason}`;
}

/** Unread alerts, shown on every screen until dismissed or opened. */
export function AlertBanner({ watch, onSelect }: { watch: Watch; onSelect: (ticker: string) => void }) {
  const qc = useQueryClient();
  const unseen = watch.events.filter((e) => !e.seen).slice(0, 3);
  if (unseen.length === 0) return null;
  const dismiss = (id: string) => api.watchSeen(id).then(() => qc.invalidateQueries({ queryKey: queryKeys.watch }));
  return (
    <div className="space-y-1">
      {unseen.map((e) => (
        <div key={e.id} className="flex items-center gap-3 flex-wrap border border-blue-300 dark:border-blue-800 bg-blue-50 dark:bg-blue-950/40 px-3 py-2 text-xs">
          <BellRing className="h-3.5 w-3.5 text-blue-600 dark:text-blue-400 shrink-0" />
          <span className="font-medium">{headline(e)}</span>
          <span className="text-muted-foreground">
            {price(e.price)} at {when(e.at)} · {EVENT_LABEL[e.status].toLowerCase()}
          </span>
          <div className="flex-1" />
          <RerunButton e={e} />
          <button className={btn} onClick={() => { dismiss(e.id); onSelect(e.ticker); }}>
            {e.status === "done" ? "View recommendation" : "View"}
          </button>
          <button className="p-1 hover:bg-accent" aria-label="Dismiss" onClick={() => dismiss(e.id)}>
            <X className="h-3.5 w-3.5" />
          </button>
        </div>
      ))}
    </div>
  );
}

/** Price alerts for one stock, on its detail page. */
export function WatchPanel({ ticker, basePrice, onSelect }: {
  ticker: string;
  basePrice: number | null;
  onSelect: (ticker: string) => void;
}) {
  const watch = useWatch();
  if (!watch.data) return null;
  const live = watch.data.prices[ticker]?.price ?? null;
  const move = basePrice && live ? (live / basePrice - 1) * 100 : null;
  const events = watch.data.events.filter((e) => e.ticker === ticker).slice(0, 5);
  const s = watch.data.settings;
  return (
    <div className="border border-border p-3 space-y-3">
      <BlockHeading title="Price alerts">
        <span className="text-[11px] text-muted-foreground font-mono tabular-nums">
          {live != null && <>Now {price(live)} <span className={changeColor(move)}>({signed(move)} since analysis)</span> · </>}
          {!s.enabled ? "watching off" : s.auto ? `auto re-run · also at ±${s.move_pct}%` : `manual re-run · also at ±${s.move_pct}%`}
        </span>
      </BlockHeading>
      <LevelEditor ticker={ticker} watch={watch.data} />
      {events.length > 0 && (
        <div className="border border-border">
          {events.map((e) => <EventRow key={e.id} e={e} onSelect={onSelect} showTicker={false} />)}
        </div>
      )}
    </div>
  );
}

function Settings({ watch }: { watch: Watch }) {
  const qc = useQueryClient();
  const save = useMutation({
    mutationFn: api.watchSettings,
    onSuccess: () => qc.invalidateQueries({ queryKey: queryKeys.watch }),
  });
  const check = useMutation({
    mutationFn: api.watchCheck,
    onSuccess: () => setTimeout(() => qc.invalidateQueries({ queryKey: queryKeys.watch }), 4000),
  });
  const s = watch.settings;
  return (
    <div className="border border-border p-3 space-y-3">
      <BlockHeading title="Price watch">
        <span className="text-[11px] text-muted-foreground">
          TSX {watch.market_open ? "open" : "closed"}
          {watch.last_check ? ` · checked ${when(watch.last_check)}` : ""}
        </span>
        <button className={btn} onClick={() => check.mutate()} disabled={check.isPending}>
          <RefreshCw className={`h-3.5 w-3.5 ${check.isPending ? "animate-spin" : ""}`} /> Check now
        </button>
      </BlockHeading>
      <div className="flex flex-wrap items-center gap-x-5 gap-y-2 text-xs">
        <label className="flex items-center gap-1.5">
          <input type="checkbox" checked={s.enabled} onChange={(e) => save.mutate({ enabled: e.target.checked })} />
          Watch prices
        </label>
        <label className="flex items-center gap-1.5">
          <input type="checkbox" checked={s.auto} onChange={(e) => save.mutate({ auto: e.target.checked })} />
          Re-run automatically (off: alert only, you click Re-run)
        </label>
        <label className="flex items-center gap-1.5">
          also alert when price moves ±
          <NumberField value={s.move_pct} onCommit={(v) => v && save.mutate({ move_pct: v })} />%
        </label>
        <label className="flex items-center gap-1.5">
          check every
          <NumberField value={s.interval_min} step="1" onCommit={(v) => v && save.mutate({ interval_min: v })} />min
        </label>
        <label className="flex items-center gap-1.5">
          at most
          <NumberField value={s.max_reruns_per_day} step="1"
            onCommit={(v) => v != null && save.mutate({ max_reruns_per_day: v })} />auto re-runs/day
        </label>
        <label className="flex items-center gap-1.5">
          <input type="checkbox" checked={s.quick} onChange={(e) => save.mutate({ quick: e.target.checked })} />
          Quick re-run (skip risk debate)
        </label>
      </div>
      {save.error && <p className="text-xs text-destructive">{(save.error as Error).message}</p>}
      <Notice label="How">
        Prices are checked while the TSX is open (Yahoo quotes can lag ~15 min). Each stock alerts at most once a day.
        A re-run uses the models from the stock&apos;s last analysis and costs about as much as a normal run; the new
        recommendation shows up here and in the banner at the top.
      </Notice>
    </div>
  );
}

function Levels({ watch, analyses }: { watch: Watch; analyses: AnalysisSummary[] }) {
  if (analyses.length === 0) return null;
  return (
    <div className="space-y-2">
      <BlockHeading title="Watched stocks" />
      <div className="border border-border">
        {analyses.map((a) => {
          const base = a.quote.price;
          const live = watch.prices[a.ticker]?.price ?? null;
          const move = base && live ? (live / base - 1) * 100 : null;
          return (
            <div key={a.ticker} className="border-t border-border first:border-t-0 px-3 py-2 flex gap-x-6 gap-y-2 flex-wrap items-start">
              <div className="w-56 shrink-0 space-y-1">
                <div className="flex items-center gap-2">
                  <span className="font-mono text-sm font-medium">{a.ticker}</span>
                  <RatingBadge rating={a.rating} />
                </div>
                <div className="text-[11px] font-mono tabular-nums text-muted-foreground">
                  rated at {price(base, a.quote.currency)} · now {price(live)}{" "}
                  <span className={changeColor(move)}>{signed(move)}</span>
                </div>
              </div>
              <div className="flex-1 min-w-[260px]"><LevelEditor ticker={a.ticker} watch={watch} /></div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function Alerts({ onSelect }: { onSelect: (ticker: string) => void }) {
  const watch = useWatch();
  const analyses = useQuery({ queryKey: queryKeys.analyses, queryFn: api.analyses });

  if (watch.isLoading) return <Skeleton />;
  if (!watch.data) return <p className="text-xs text-destructive">{(watch.error as Error)?.message}</p>;

  return (
    <div className="space-y-4">
      <Settings watch={watch.data} />
      <Levels watch={watch.data} analyses={analyses.data?.analyses ?? []} />
      <div className="space-y-2">
        <BlockHeading title="Alerts" />
        {watch.data.events.length === 0 ? (
          <EmptyState icon={BellRing} message="No price alerts yet." />
        ) : (
          <div className="border border-border">
            {watch.data.events.map((e) => <EventRow key={e.id} e={e} onSelect={onSelect} />)}
          </div>
        )}
      </div>
    </div>
  );
}
