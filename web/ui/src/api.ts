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

export const api = {
  universe: (refresh = false) => request<Universe>(`/api/universe${refresh ? "?refresh=true" : ""}`),
  analyses: () => request<{ analyses: AnalysisSummary[] }>("/api/analyses"),
  analysis: (ticker: string, date?: string) =>
    request<Analysis>(`/api/analyses/${encodeURIComponent(ticker)}${date ? `?date=${date}` : ""}`),
  runs: () => request<{ runs: Run[] }>("/api/runs"),
  createRun: (body: { tickers: { ticker: string; company?: string }[]; trade_date?: string; quick: boolean }) =>
    request<Run>("/api/runs", { method: "POST", body: JSON.stringify(body) }),
  cancelRun: (id: string) => request<Run>(`/api/runs/${id}/cancel`, { method: "POST" }),
};

export const queryKeys = {
  universe: ["universe"] as const,
  analyses: ["analyses"] as const,
  analysis: (ticker: string, date?: string) => ["analysis", ticker, date ?? "latest"] as const,
  runs: ["runs"] as const,
};
