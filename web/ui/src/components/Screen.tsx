import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ListFilter, TrendingUp } from "lucide-react";
import { api, queryKeys, type AnalysisSummary, type Rating } from "../api";
import {
  EmptyState,
  Notice,
  RatingBadge,
  RatingTrail,
  Segmented,
  Skeleton,
  changeColor,
  daysSince,
  duration,
  price,
  signed,
} from "./shared";

type Filter = Rating | "all";

const FILTERS: { key: Filter; label: string }[] = [
  { key: "all", label: "All" },
  { key: "BUY", label: "Buy" },
  { key: "OVERWEIGHT", label: "Overweight" },
  { key: "HOLD", label: "Hold" },
  { key: "UNDERWEIGHT", label: "Underweight" },
  { key: "SELL", label: "Sell" },
];

const RATING_ORDER: Record<Rating, number> = { BUY: 0, OVERWEIGHT: 1, HOLD: 2, UNDERWEIGHT: 3, SELL: 4, UNKNOWN: 5 };

// Analyses older than this describe a market that has moved on.
const STALE_DAYS = 7;

function Row({ a, onSelect }: { a: AnalysisSummary; onSelect: (ticker: string) => void }) {
  const age = daysSince(a.trade_date);
  return (
    <tr
      onClick={() => onSelect(a.ticker)}
      className="border-t border-border hover:bg-accent/30 transition-colors cursor-pointer"
    >
      <td className="px-3 py-2">
        <div className="font-mono text-sm font-medium">{a.ticker}</div>
        <div className="text-[11px] text-muted-foreground truncate max-w-[200px]">{a.company ?? "—"}</div>
      </td>
      <td className="px-3 py-2">
        <RatingBadge rating={a.rating} />
      </td>
      <td className="px-3 py-2 hidden md:table-cell">
        <RatingTrail history={a.history} />
      </td>
      <td className="px-3 py-2 text-xs font-mono tabular-nums hidden sm:table-cell">
        {price(a.quote.price, a.quote.currency)}
      </td>
      <td className={`px-3 py-2 text-xs font-mono tabular-nums hidden sm:table-cell ${changeColor(a.quote.change_pct)}`}>
        {signed(a.quote.change_pct)}
      </td>
      <td className="px-3 py-2 text-xs font-mono tabular-nums">
        <span className={age > STALE_DAYS ? "text-destructive" : ""} title={age > STALE_DAYS ? "Stale — re-run before acting" : undefined}>
          {a.trade_date}
        </span>
        <span className="text-[10px] text-muted-foreground ml-1.5">{age === 0 ? "today" : `${age}d`}</span>
      </td>
      <td className="px-3 py-2 text-xs font-mono tabular-nums text-muted-foreground hidden lg:table-cell">
        {duration(a.duration_s)}
      </td>
      <td className="px-3 py-2 text-[11px] text-muted-foreground hidden xl:table-cell truncate max-w-[160px]">
        {a.model.deep}
      </td>
    </tr>
  );
}

export function Screen({ onSelect }: { onSelect: (ticker: string) => void }) {
  const [filter, setFilter] = useState<Filter>("all");
  const { data, isLoading } = useQuery({ queryKey: queryKeys.analyses, queryFn: api.analyses });

  if (isLoading) return <Skeleton />;

  const all = data?.analyses ?? [];
  if (all.length === 0) {
    return <EmptyState icon={TrendingUp} message="No analyses yet. Pick stocks under Run to start one." />;
  }

  const rows = all
    .filter((a) => filter === "all" || a.rating === filter)
    .sort((x, y) => RATING_ORDER[x.rating] - RATING_ORDER[y.rating] || x.ticker.localeCompare(y.ticker));
  const stale = all.filter((a) => daysSince(a.trade_date) > STALE_DAYS).length;

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <Segmented options={FILTERS} value={filter} onChange={setFilter} />
        <div className="text-xs text-muted-foreground flex items-center gap-1.5">
          <ListFilter className="h-3.5 w-3.5" />
          {rows.length} of {all.length} shown
          {stale > 0 && <span className="text-destructive">· {stale} older than {STALE_DAYS} days</span>}
        </div>
      </div>

      <Notice label="Not advice">
        Each rating is one LLM&apos;s opinion after a scripted debate. It can misread numbers or rely on stale
        news. Open a ticker and read the bear case and the risk debate before you put money behind a Buy.
      </Notice>

      <div className="border border-border overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-accent/20 text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
              <th className="px-3 py-2 text-left font-medium">Ticker</th>
              <th className="px-3 py-2 text-left font-medium">Rating</th>
              <th className="px-3 py-2 text-left font-medium hidden md:table-cell">Trend</th>
              <th className="px-3 py-2 text-left font-medium hidden sm:table-cell">Price</th>
              <th className="px-3 py-2 text-left font-medium hidden sm:table-cell">Day</th>
              <th className="px-3 py-2 text-left font-medium">As of</th>
              <th className="px-3 py-2 text-left font-medium hidden lg:table-cell">Took</th>
              <th className="px-3 py-2 text-left font-medium hidden xl:table-cell">Model</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((a) => (
              <Row key={a.ticker} a={a} onSelect={onSelect} />
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
