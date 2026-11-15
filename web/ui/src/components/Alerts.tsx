import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowRight, BellRing, RefreshCw } from "lucide-react";
import { api, queryKeys, type AnalysisSummary, type Watch, type WatchEvent, type WatchLevel } from "../api";
import { BlockHeading, EmptyState, Notice, RatingBadge, Skeleton, btn, changeColor, price, signed } from "./shared";

const EVENT_LABEL: Record<WatchEvent["status"], string> = {
  waiting: "Waiting for the runner",
  running: "Re-rating…",
  done: "Re-rated",
  failed: "Re-rate failed",
  skipped: "Not re-rated (daily limit)",
};

function when(ts: number): string {
  return new Date(ts * 1000).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
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
          Re-rate automatically
        </label>
        <label className="flex items-center gap-1.5">
          when price moves ±
          <NumberField value={s.move_pct} onCommit={(v) => v && save.mutate({ move_pct: v })} />%
        </label>
        <label className="flex items-center gap-1.5">
          check every
          <NumberField value={s.interval_min} step="1" onCommit={(v) => v && save.mutate({ interval_min: v })} />min
        </label>
        <label className="flex items-center gap-1.5">
          at most
          <NumberField value={s.max_reruns_per_day} step="1"
            onCommit={(v) => v != null && save.mutate({ max_reruns_per_day: v })} />re-rates/day
        </label>
        <label className="flex items-center gap-1.5">
          <input type="checkbox" checked={s.quick} onChange={(e) => save.mutate({ quick: e.target.checked })} />
          Quick re-rate (skip risk debate)
        </label>
      </div>
      {save.error && <p className="text-xs text-destructive">{(save.error as Error).message}</p>}
      <Notice label="How">
        Prices are checked while the TSX is open (Yahoo quotes can lag ~15 min). A stock is re-rated at most once a
        day, using the models from its last analysis; each re-rate costs about as much as a normal run. You get a
        macOS notification when it triggers and when the new rating is in.
      </Notice>
    </div>
  );
}

function Levels({ watch, analyses }: { watch: Watch; analyses: AnalysisSummary[] }) {
  const qc = useQueryClient();
  const save = useMutation({
    mutationFn: ({ ticker, level }: { ticker: string; level: WatchLevel }) => api.watchLevels(ticker, level),
    onSuccess: (data) => qc.setQueryData(queryKeys.watch, data),
  });
  if (analyses.length === 0) return null;
  return (
    <div className="space-y-2">
      <BlockHeading title="Watched stocks" />
      <div className="border border-border overflow-x-auto">
        <table className="w-full text-xs">
          <thead className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
            <tr>
              <th className="text-left px-3 py-2 font-medium">Stock</th>
              <th className="text-left px-3 py-2 font-medium">Rating</th>
              <th className="text-right px-3 py-2 font-medium">Rated at</th>
              <th className="text-right px-3 py-2 font-medium">Now</th>
              <th className="text-right px-3 py-2 font-medium">Move</th>
              <th className="text-left px-3 py-2 font-medium">Re-rate above</th>
              <th className="text-left px-3 py-2 font-medium">Re-rate below</th>
            </tr>
          </thead>
          <tbody>
            {analyses.map((a) => {
              const base = a.quote.price;
              const live = watch.prices[a.ticker]?.price ?? null;
              const move = base && live ? (live / base - 1) * 100 : null;
              const level = watch.levels[a.ticker] ?? { above: null, below: null };
              const set = (patch: Partial<WatchLevel>) => save.mutate({ ticker: a.ticker, level: { ...level, ...patch } });
              return (
                <tr key={a.ticker} className="border-t border-border">
                  <td className="px-3 py-1.5 font-mono font-medium">{a.ticker}</td>
                  <td className="px-3 py-1.5"><RatingBadge rating={a.rating} /></td>
                  <td className="px-3 py-1.5 text-right font-mono tabular-nums">{price(base, a.quote.currency)}</td>
                  <td className="px-3 py-1.5 text-right font-mono tabular-nums">{price(live, a.quote.currency)}</td>
                  <td className={`px-3 py-1.5 text-right font-mono tabular-nums ${changeColor(move)}`}>{signed(move)}</td>
                  <td className="px-3 py-1.5"><NumberField value={level.above} placeholder="—" onCommit={(v) => set({ above: v })} /></td>
                  <td className="px-3 py-1.5"><NumberField value={level.below} placeholder="—" onCommit={(v) => set({ below: v })} /></td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {save.error && <p className="text-xs text-destructive">{(save.error as Error).message}</p>}
    </div>
  );
}

function Event({ e, onSelect }: { e: WatchEvent; onSelect: (ticker: string) => void }) {
  const changed = e.new_rating && e.new_rating !== e.prev_rating;
  return (
    <button
      onClick={() => onSelect(e.ticker)}
      className={`w-full flex items-center gap-3 px-3 py-2 text-left border-t border-border first:border-t-0 hover:bg-accent/30 flex-wrap ${
        e.seen ? "" : "bg-blue-50 dark:bg-blue-950/30"
      }`}
    >
      <span className="text-xs font-mono tabular-nums w-28 shrink-0">{when(e.at)}</span>
      <span className="font-mono text-sm font-medium w-20">{e.ticker}</span>
      <span className="text-xs text-muted-foreground flex-1 min-w-[180px]">
        {e.reason} · {price(e.base_price)} → {price(e.price)}
      </span>
      <span className="inline-flex items-center gap-1.5">
        <RatingBadge rating={e.prev_rating} />
        {e.new_rating && <><ArrowRight className="h-3 w-3" /><RatingBadge rating={e.new_rating} /></>}
      </span>
      <span className={`text-[11px] w-40 text-right ${changed ? "font-medium" : "text-muted-foreground"} ${
        e.status === "failed" ? "text-destructive" : ""}`} title={e.error}>
        {changed ? "Rating changed" : EVENT_LABEL[e.status]}
      </span>
    </button>
  );
}

export function Alerts({ onSelect }: { onSelect: (ticker: string) => void }) {
  const qc = useQueryClient();
  const watch = useQuery({ queryKey: queryKeys.watch, queryFn: api.watch, refetchInterval: 30_000 });
  const analyses = useQuery({ queryKey: queryKeys.analyses, queryFn: api.analyses });

  const unseen = watch.data?.unseen ?? 0;
  useEffect(() => {
    // Opening the tab counts as reading the alerts.
    if (unseen > 0) api.watchSeen().then(() => qc.invalidateQueries({ queryKey: queryKeys.watch }));
  }, [unseen, qc]);

  if (watch.isLoading) return <Skeleton />;
  if (!watch.data) return <p className="text-xs text-destructive">{(watch.error as Error)?.message}</p>;

  return (
    <div className="space-y-4">
      <Settings watch={watch.data} />
      <Levels watch={watch.data} analyses={analyses.data?.analyses ?? []} />
      <div className="space-y-2">
        <BlockHeading title="Alerts" />
        {watch.data.events.length === 0 ? (
          <EmptyState icon={BellRing} message="No price triggers yet." />
        ) : (
          <div className="border border-border">
            {watch.data.events.map((e) => <Event key={e.id} e={e} onSelect={onSelect} />)}
          </div>
        )}
      </div>
    </div>
  );
}
