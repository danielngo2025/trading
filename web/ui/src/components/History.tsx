import { useState } from "react";
import { History as HistoryIcon } from "lucide-react";
import type { Run } from "../api";
import { RatingBadge, StatusBadge, EmptyState, duration } from "./shared";
import { ActiveRun } from "./RunPanel";

function when(ts: number): string {
  return new Date(ts * 1000).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

export function History({ runs, onSelect }: { runs: Run[]; onSelect: (ticker: string) => void }) {
  const [open, setOpen] = useState<string | null>(null);

  if (runs.length === 0) return <EmptyState icon={HistoryIcon} message="No runs yet." />;

  return (
    <div className="border border-border">
      {runs.map((run) => {
        const done = run.items.filter((i) => i.status === "done");
        return (
          <div key={run.id} className="border-t border-border first:border-t-0">
            <button
              onClick={() => setOpen(open === run.id ? null : run.id)}
              className="w-full flex items-center gap-3 px-3 py-2 text-left hover:bg-accent/30 transition-colors flex-wrap"
            >
              <span className="text-xs font-mono tabular-nums w-28">{when(run.created_at)}</span>
              <span className="text-xs text-muted-foreground font-mono">as of {run.trade_date}</span>
              <span className="text-xs text-muted-foreground">
                {done.length}/{run.items.length} stocks{run.quick ? " · quick" : ""}
              </span>
              <div className="flex-1 flex gap-1 flex-wrap justify-end">
                {done.slice(0, 10).map((i) => i.rating && (
                  <span key={i.ticker} className="inline-flex items-center gap-1 text-[10px] font-mono">
                    {i.ticker.replace(/\.TO$/, "")}
                    <RatingBadge rating={i.rating} />
                  </span>
                ))}
              </div>
              <StatusBadge status={run.status} />
            </button>
            {open === run.id && (
              <div className="px-3 pb-3">
                <ActiveRun run={run} onSelect={onSelect} />
                {run.items[0]?.started_at && run.items.at(-1)?.finished_at && (
                  <p className="text-[11px] text-muted-foreground mt-1.5">
                    Total {duration(Math.round(run.items.at(-1)!.finished_at! - run.items[0].started_at!))}
                  </p>
                )}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
