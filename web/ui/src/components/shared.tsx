import type { LucideIcon } from "lucide-react";
import type { ItemStatus, Rating } from "../api";

export const RATING_LABELS: Record<Rating, string> = {
  BUY: "Buy",
  OVERWEIGHT: "Overweight",
  HOLD: "Hold",
  UNDERWEIGHT: "Underweight",
  SELL: "Sell",
  UNKNOWN: "Unclear",
};

const RATING_CLASSES: Record<Rating, string> = {
  BUY: "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300",
  OVERWEIGHT: "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/30 dark:text-emerald-300",
  HOLD: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300",
  UNDERWEIGHT: "bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-300",
  SELL: "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300",
  UNKNOWN: "bg-muted text-muted-foreground",
};

const RATING_DOTS: Record<Rating, string> = {
  BUY: "bg-green-500",
  OVERWEIGHT: "bg-emerald-400",
  HOLD: "bg-yellow-400",
  UNDERWEIGHT: "bg-orange-400",
  SELL: "bg-red-500",
  UNKNOWN: "bg-muted-foreground/40",
};

export function RatingBadge({ rating }: { rating: Rating }) {
  return (
    <span
      className={`inline-flex items-center px-1.5 py-0.5 text-[10px] font-medium whitespace-nowrap ${RATING_CLASSES[rating]}`}
    >
      {RATING_LABELS[rating]}
    </span>
  );
}

/** Oldest → newest, so a drift from Hold to Sell reads left to right. */
export function RatingTrail({ history }: { history: { trade_date: string; rating: Rating | null }[] }) {
  return (
    <div className="flex items-center gap-1">
      {[...history].reverse().map((h) => (
        <span
          key={h.trade_date}
          title={`${h.trade_date}: ${RATING_LABELS[h.rating ?? "UNKNOWN"]}`}
          className={`h-2 w-2 rounded-full ${RATING_DOTS[h.rating ?? "UNKNOWN"]}`}
        />
      ))}
    </div>
  );
}

const STATUS_CLASSES: Record<ItemStatus, string> = {
  queued: "bg-muted text-muted-foreground",
  running: "bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300",
  done: "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300",
  failed: "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300",
  cancelled: "bg-muted text-muted-foreground",
  interrupted: "bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-300",
};

export function StatusBadge({ status }: { status: ItemStatus }) {
  return (
    <span
      className={`inline-flex items-center px-1.5 py-0.5 text-[10px] font-medium capitalize whitespace-nowrap ${STATUS_CLASSES[status]}`}
    >
      {status}
    </span>
  );
}

export function price(value: number | null, currency?: string | null): string {
  if (value == null) return "—";
  return `$${value.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}${
    currency && currency !== "CAD" ? ` ${currency}` : ""
  }`;
}

export function marketCap(value: number | null): string {
  if (value == null) return "—";
  if (value >= 1e12) return `$${(value / 1e12).toFixed(2)}T`;
  if (value >= 1e9) return `$${(value / 1e9).toFixed(1)}B`;
  return `$${(value / 1e6).toFixed(0)}M`;
}

export function signed(value: number | null, digits = 1): string {
  if (value == null) return "—";
  return `${value > 0 ? "+" : ""}${value.toFixed(digits)}%`;
}

export function changeColor(value: number | null): string {
  if (value == null || value === 0) return "text-muted-foreground";
  return value > 0 ? "text-green-600 dark:text-green-500" : "text-destructive";
}

export function daysSince(isoDate: string): number {
  return Math.floor((Date.now() - new Date(`${isoDate}T00:00:00`).getTime()) / 86_400_000);
}

export function duration(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  const m = Math.floor(seconds / 60);
  return m < 60 ? `${m}m ${seconds % 60}s` : `${Math.floor(m / 60)}h ${m % 60}m`;
}

export function BlockHeading({ title, children }: { title: string; children?: React.ReactNode }) {
  return (
    <div className="flex items-center gap-3 flex-wrap">
      <h4 className="text-[11px] font-mono uppercase tracking-widest text-muted-foreground font-medium">
        {title}
      </h4>
      <div className="flex-1" />
      {children}
    </div>
  );
}

export function Notice({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-start gap-3 border border-border bg-muted px-3 py-2 text-xs">
      <span className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground shrink-0 pt-0.5">
        {label}
      </span>
      <span>{children}</span>
    </div>
  );
}

export function EmptyState({ icon: Icon, message }: { icon: LucideIcon; message: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-16 text-center">
      <div className="bg-muted/50 p-4 mb-4">
        <Icon className="h-10 w-10 text-muted-foreground/50" />
      </div>
      <p className="text-sm text-muted-foreground">{message}</p>
    </div>
  );
}

export function Skeleton() {
  return (
    <div className="space-y-2 animate-pulse">
      {Array.from({ length: 6 }, (_, i) => (
        <div key={i} className="h-9 bg-muted" />
      ))}
    </div>
  );
}

export function Segmented<T extends string>({
  options,
  value,
  onChange,
}: {
  options: { key: T; label: string }[];
  value: T;
  onChange: (next: T) => void;
}) {
  return (
    <div className="flex items-center gap-1 border border-border rounded-md p-0.5 flex-wrap">
      {options.map((o) => (
        <button
          key={o.key}
          onClick={() => onChange(o.key)}
          className={`px-2 py-1 text-xs rounded transition-colors ${
            value === o.key
              ? "bg-accent text-foreground font-medium"
              : "text-muted-foreground hover:text-foreground hover:bg-accent/50"
          }`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

/** Line-variant tabs, matching nexio-per's PageTabBar. */
export function TabBar<T extends string>({
  items,
  value,
  onChange,
}: {
  items: { value: T; label: React.ReactNode }[];
  value: T;
  onChange: (next: T) => void;
}) {
  return (
    <div className="flex items-center gap-1 h-9 border-b border-border">
      {items.map((item) => (
        <button
          key={item.value}
          onClick={() => onChange(item.value)}
          className={`relative inline-flex h-full items-center gap-1.5 px-2 text-sm font-medium whitespace-nowrap transition-colors ${
            value === item.value
              ? "text-foreground after:absolute after:inset-x-0 after:-bottom-px after:h-0.5 after:bg-foreground"
              : "text-foreground/60 hover:text-foreground dark:text-muted-foreground dark:hover:text-foreground"
          }`}
        >
          {item.label}
        </button>
      ))}
    </div>
  );
}

export const btn =
  "inline-flex items-center gap-1.5 px-3 h-8 text-xs font-medium border border-border transition-colors disabled:opacity-50 disabled:pointer-events-none";
export const btnPrimary = `${btn} bg-primary text-primary-foreground border-primary hover:opacity-90`;
export const btnGhost = `${btn} hover:bg-accent`;
