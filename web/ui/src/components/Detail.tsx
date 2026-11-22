import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { WatchPanel } from "./Alerts";
import { MarketPanel } from "./Charts";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { ArrowLeft } from "lucide-react";
import { api, queryKeys, type ReportKey } from "../api";
import {
  BlockHeading,
  Notice,
  RatingBadge,
  Segmented,
  Skeleton,
  TabBar,
  changeColor,
  cost,
  daysSince,
  duration,
  marketCap,
  price,
  signed,
  tokens,
} from "./shared";

type Section = "analysts" | "research" | "trader" | "risk";

const SECTIONS: { value: Section; label: string; reports: { key: ReportKey; label: string }[] }[] = [
  {
    value: "analysts",
    label: "Analysts",
    reports: [
      { key: "market", label: "Technical" },
      { key: "fundamentals", label: "Fundamentals" },
      { key: "news", label: "News" },
      { key: "sentiment", label: "Sentiment" },
    ],
  },
  {
    value: "research",
    label: "Bull vs bear",
    reports: [
      { key: "bear", label: "Bear case" },
      { key: "bull", label: "Bull case" },
      { key: "research_manager", label: "Manager verdict" },
    ],
  },
  { value: "trader", label: "Trader", reports: [{ key: "trader", label: "Trade plan" }] },
  {
    value: "risk",
    label: "Risk debate",
    reports: [
      { key: "conservative", label: "Conservative" },
      { key: "neutral", label: "Neutral" },
      { key: "aggressive", label: "Aggressive" },
    ],
  },
];

function Markdown({ text }: { text: string }) {
  if (!text?.trim()) return <p className="text-xs text-muted-foreground py-4">Nothing recorded for this step.</p>;
  return (
    <div className="prose prose-sm dark:prose-invert max-w-none prose-headings:font-medium prose-table:text-xs">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{text}</ReactMarkdown>
    </div>
  );
}

function ReportSection({ section, reports }: { section: (typeof SECTIONS)[number]; reports: Record<ReportKey, string> }) {
  const [active, setActive] = useState<ReportKey>(section.reports[0].key);
  const current = section.reports.some((r) => r.key === active) ? active : section.reports[0].key;
  return (
    <div className="space-y-3">
      {section.reports.length > 1 && (
        <Segmented options={section.reports} value={current} onChange={setActive} />
      )}
      <div className="border border-border px-4 py-3">
        <Markdown text={reports[current]} />
      </div>
    </div>
  );
}

export function Detail({ ticker, onBack, onSelect }: {
  ticker: string;
  onBack: () => void;
  onSelect: (ticker: string) => void;
}) {
  const [date, setDate] = useState<string | undefined>();
  const [section, setSection] = useState<Section>("analysts");
  const { data, isLoading, error } = useQuery({
    queryKey: queryKeys.analysis(ticker, date),
    queryFn: () => api.analysis(ticker, date),
  });

  const back = (
    <button onClick={onBack} className="inline-flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground">
      <ArrowLeft className="h-3.5 w-3.5" /> All stocks
    </button>
  );

  if (isLoading) return <div className="space-y-4">{back}<Skeleton /></div>;
  if (error || !data) return <div className="space-y-4">{back}<Notice label="Error">{(error as Error)?.message ?? "Not found"}</Notice></div>;

  const age = daysSince(data.trade_date);
  const stats = [
    { label: "Price now", value: price(data.quote.price, data.quote.currency) },
    { label: "Day", value: signed(data.quote.change_pct), className: changeColor(data.quote.change_pct) },
    { label: "Mkt cap", value: marketCap(data.quote.market_cap) },
    { label: "As of", value: age === 0 ? data.trade_date : `${data.trade_date} · ${age}d`, className: age > 7 ? "text-destructive" : "" },
    { label: "Took", value: duration(data.duration_s) },
    {
      label: "Tokens",
      value: data.tokens ? `${tokens(data.tokens)} · ${data.tokens.calls} calls${cost(data.tokens.cost_usd)}` : "—",
    },
    {
      label: "Model",
      // Older analyses predate the separate analyst model.
      value: [...new Set([data.model.deep, data.model.quick, data.model.analyst ?? data.model.quick])].join(" / "),
    },
  ];
  const active = SECTIONS.find((s) => s.value === section)!;

  return (
    <div className="space-y-4">
      {back}

      <div className="flex items-center gap-3 flex-wrap">
        <h2 className="font-mono text-xl font-medium">{data.ticker}</h2>
        <span className="text-sm text-muted-foreground">{data.company}</span>
        <RatingBadge rating={data.rating} />
        <div className="flex-1" />
        {data.dates.length > 1 && (
          <select
            value={date ?? data.trade_date}
            onChange={(e) => setDate(e.target.value)}
            className="h-8 px-2 text-xs font-mono border border-border bg-background focus:outline-none focus:ring-1 focus:ring-ring"
          >
            {data.dates.map((d) => (
              <option key={d} value={d}>{d}</option>
            ))}
          </select>
        )}
      </div>

      <div className="border border-border grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-px bg-border">
        {stats.map((s) => (
          <div key={s.label} className="bg-background px-3 py-2 min-w-0">
            <div className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground">{s.label}</div>
            <div className={`text-sm font-mono tabular-nums truncate ${s.className ?? ""}`} title={s.value}>{s.value}</div>
          </div>
        ))}
      </div>

      {data.trade_date === data.dates[0] && (
        <WatchPanel ticker={data.ticker} basePrice={data.quote.price} onSelect={onSelect} />
      )}

      {data.warnings.length > 0 && (
        <div className="border border-destructive/40 bg-destructive/5 px-3 py-2 text-xs space-y-1">
          <div className="font-mono text-[10px] uppercase tracking-widest text-destructive">
            Data gaps — don&apos;t act on this rating
          </div>
          <ul className="list-disc pl-4 space-y-0.5">
            {data.warnings.map((w) => <li key={w}>{w}</li>)}
          </ul>
          <p className="text-muted-foreground">
            The agents still produced a confident decision, but it rests on missing or wrong inputs. Analyse it again
            from the Run tab; results with gaps are redone automatically, no Force re-run needed. If the same gap comes
            back, try a stronger quick model or check the figures yourself.
          </p>
        </div>
      )}

      <MarketPanel ticker={data.ticker} />

      <div className="space-y-2">
        <BlockHeading title="Portfolio manager decision" />
        <div className="border border-border px-4 py-3 bg-muted/30">
          <Markdown text={data.reports.final} />
        </div>
      </div>

      {data.reused?.fundamentals && (
        <Notice label="Reused">
          The fundamentals report is from the {data.reused.fundamentals} analysis, not re-run for this date.
        </Notice>
      )}

      {age > 7 && (
        <Notice label="Stale">
          This analysis is {age} days old. Prices and news have moved since; re-run it before trading on it.
        </Notice>
      )}

      <div className="space-y-3">
        <TabBar
          items={SECTIONS.map((s) => ({ value: s.value, label: s.label }))}
          value={section}
          onChange={setSection}
        />
        <ReportSection key={`${section}-${data.trade_date}`} section={active} reports={data.reports} />
      </div>
    </div>
  );
}
