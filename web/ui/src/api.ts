export type Rating = "BUY" | "OVERWEIGHT" | "HOLD" | "UNDERWEIGHT" | "SELL" | "UNKNOWN";

export interface Quote {
  market_cap: number | null;
  price: number | null;
  change_pct: number | null;
  currency: string | null;
}

export interface UniverseStock extends Quote {
  ticker: string;
  company: string;
  sector: string;
  rank: number;
}

export interface Universe {
  name: string;
  fetched_at: number;
  stocks: UniverseStock[];
}

export interface Tokens {
  input: number;
  output: number;
  calls: number;
}

export interface ModelOption {
  value: string;
  label: string;
}

export interface Provider {
  key: string;
  label: string;
  available: boolean;
  deep: ModelOption[];
  quick: ModelOption[];
}

export interface Models {
  providers: Provider[];
  default: { provider: string; deep: string; quick: string };
}

export type ReportKey =
  | "market" | "sentiment" | "news" | "fundamentals"
  | "bull" | "bear" | "research_manager" | "trader"
  | "aggressive" | "conservative" | "neutral" | "final";

export interface AnalysisSummary {
  ticker: string;
  company: string | null;
  trade_date: string;
  rating: Rating;
  run_id: string;
  created_at: number;
  duration_s: number;
  model: { provider: string; deep: string; quick: string };
  quote: Quote;
  tokens?: Tokens;
  reused?: { fundamentals?: string };
  warnings: string[];
  history: { trade_date: string; rating: Rating | null }[];
}

export interface Analysis extends Omit<AnalysisSummary, "history"> {
  reports: Record<ReportKey, string>;
  dates: string[];
}

export type ItemStatus = "queued" | "running" | "done" | "failed" | "cancelled" | "interrupted";

export interface RunItem {
  ticker: string;
  company: string | null;
  status: ItemStatus;
  stage: string | null;
  stages_done: number;
  rating?: Rating;
  tokens?: Tokens;
  cached?: boolean;
  error?: string;
  started_at?: number;
  finished_at?: number;
}

export interface Run {
  id: string;
  created_at: number;
  trade_date: string;
  status: "queued" | "running" | "done" | "failed" | "cancelled" | "interrupted";
  quick: boolean;
  force?: boolean;
  reuse_fundamentals?: boolean;
  error?: string;
  cancel_requested?: boolean;
  config: { llm_provider: string; deep_think_llm: string; quick_think_llm: string };
  items: RunItem[];
  stages: { key: string; label: string }[];
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `${res.status} ${res.statusText}`);
  }
  return res.json();
}

export interface WatchSettings {
  enabled: boolean;
  auto: boolean;
  interval_min: number;
  move_pct: number;
  max_reruns_per_day: number;
  quick: boolean;
}

export interface WatchLevel {
  above: number | null;
  below: number | null;
}

export interface WatchEvent {
  id: string;
  ticker: string;
  at: number;
  price: number;
  base_price: number;
  base_date: string;
  reason: string;
  prev_rating: Rating;
  new_rating: Rating | null;
  status: "triggered" | "waiting" | "running" | "done" | "failed" | "skipped";
  error?: string;
  seen: boolean;
}

export interface Watch {
  settings: WatchSettings;
  levels: Record<string, WatchLevel>;
  prices: Record<string, { price: number; at: number; base: number }>;
  events: WatchEvent[];
  unseen: number;
  last_check?: number;
  market_open: boolean;
  suggestions: Record<string, { price: number; side: "above" | "below"; label: string }[]>;
}

const put = (body: unknown) => ({ method: "PUT", body: JSON.stringify(body) });

export const api = {
  universe: (refresh = false) => request<Universe>(`/api/universe${refresh ? "?refresh=true" : ""}`),
  analyses: () => request<{ analyses: AnalysisSummary[] }>("/api/analyses"),
  analysis: (ticker: string, date?: string) =>
    request<Analysis>(`/api/analyses/${encodeURIComponent(ticker)}${date ? `?date=${date}` : ""}`),
  runs: () => request<{ runs: Run[] }>("/api/runs"),
  models: () => request<Models>("/api/models"),
  createRun: (body: {
    tickers: { ticker: string; company?: string }[];
    trade_date?: string;
    quick: boolean;
    provider: string;
    deep_model: string;
    quick_model: string;
    force: boolean;
    reuse_fundamentals: boolean;
  }) =>
    request<Run>("/api/runs", { method: "POST", body: JSON.stringify(body) }),
  cancelRun: (id: string) => request<Run>(`/api/runs/${id}/cancel`, { method: "POST" }),
  watch: () => request<Watch>("/api/watch"),
  watchSettings: (body: Partial<WatchSettings>) => request<WatchSettings>("/api/watch/settings", put(body)),
  watchLevels: (ticker: string, body: WatchLevel) =>
    request<Watch>(`/api/watch/levels/${encodeURIComponent(ticker)}`, put(body)),
  watchSeen: (eventId?: string) =>
    request<{ ok: boolean }>(`/api/watch/seen${eventId ? `?event_id=${eventId}` : ""}`, { method: "POST" }),
  watchRerun: (eventId: string) => request<WatchEvent>(`/api/watch/events/${eventId}/rerun`, { method: "POST" }),
  watchCheck: () => request<{ ok: boolean }>("/api/watch/check", { method: "POST" }),
};

export const queryKeys = {
  universe: ["universe"] as const,
  analyses: ["analyses"] as const,
  analysis: (ticker: string, date?: string) => ["analysis", ticker, date ?? "latest"] as const,
  runs: ["runs"] as const,
  models: ["models"] as const,
  watch: ["watch"] as const,
};
