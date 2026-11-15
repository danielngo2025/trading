import { useEffect, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Moon, Sun, TrendingUp } from "lucide-react";
import { api, queryKeys } from "./api";
import { TabBar } from "./components/shared";
import { Screen } from "./components/Screen";
import { RunPanel } from "./components/RunPanel";
import { History } from "./components/History";
import { Alerts } from "./components/Alerts";
import { Detail } from "./components/Detail";

type Tab = "screen" | "run" | "history" | "alerts";

function useTheme() {
  const [dark, setDark] = useState(() => {
    try {
      const saved = localStorage.getItem("theme");
      return saved ? saved === "dark" : window.matchMedia("(prefers-color-scheme: dark)").matches;
    } catch {
      return false;
    }
  });
  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
    try {
      localStorage.setItem("theme", dark ? "dark" : "light");
    } catch {
      /* private mode */
    }
  }, [dark]);
  return [dark, setDark] as const;
}

export function App() {
  const qc = useQueryClient();
  const [tab, setTab] = useState<Tab>("screen");
  const [ticker, setTicker] = useState<string | null>(null);
  const [dark, setDark] = useTheme();

  const runs = useQuery({
    queryKey: queryKeys.runs,
    queryFn: api.runs,
    // Poll only while something is in flight.
    refetchInterval: (q) =>
      q.state.data?.runs.some((r) => r.status === "queued" || r.status === "running") ? 3000 : false,
  });
  const watch = useQuery({ queryKey: queryKeys.watch, queryFn: api.watch, refetchInterval: 60_000 });
  const unseen = watch.data?.unseen ?? 0;
  const latest = runs.data?.runs[0] ?? null;
  const running = latest && (latest.status === "queued" || latest.status === "running") ? latest : null;
  const doneCount = latest?.items.filter((i) => i.status === "done").length ?? 0;

  // Each finished stock should show up on the Screen without a manual refresh.
  useEffect(() => {
    qc.invalidateQueries({ queryKey: queryKeys.analyses });
  }, [doneCount, latest?.status, qc]);

  // The watcher starts runs on its own; pick them up when it reports one in flight.
  const watcherBusy = watch.data?.events.some((e) => e.status === "running") ?? false;
  useEffect(() => {
    qc.invalidateQueries({ queryKey: queryKeys.runs });
  }, [watcherBusy, qc]);

  return (
    <div className="min-h-screen">
      <header className="border-b border-border">
        <div className="max-w-6xl mx-auto px-4 h-12 flex items-center gap-2">
          <TrendingUp className="h-4 w-4" />
          <span className="text-sm font-medium">TradingAgents</span>
          <span className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground">S&amp;P/TSX 60</span>
          <div className="flex-1" />
          {running && (
            <button onClick={() => { setTicker(null); setTab("run"); }} className="text-xs text-blue-600 dark:text-blue-400 font-mono">
              Running {running.items.filter((i) => i.status === "done").length}/{running.items.length}
            </button>
          )}
          <button onClick={() => setDark(!dark)} className="p-1.5 hover:bg-accent" aria-label="Toggle theme">
            {dark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
          </button>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 py-4">
        {ticker ? (
          <Detail ticker={ticker} onBack={() => setTicker(null)} />
        ) : (
          <div className="space-y-4">
            <TabBar
              items={[
                { value: "screen", label: "Screen" },
                { value: "run", label: running ? <>Run <span className="h-1.5 w-1.5 rounded-full bg-blue-500 animate-pulse" /></> : "Run" },
                { value: "history", label: "History" },
                {
                  value: "alerts",
                  label: unseen > 0 && tab !== "alerts"
                    ? <>Alerts <span className="text-[10px] font-mono px-1 bg-blue-600 text-white">{unseen}</span></>
                    : "Alerts",
                },
              ]}
              value={tab}
              onChange={setTab}
            />
            {tab === "screen" && <Screen onSelect={setTicker} />}
            {tab === "run" && <RunPanel activeRun={running ?? latest} onSelect={setTicker} />}
            {tab === "history" && <History runs={runs.data?.runs ?? []} onSelect={setTicker} />}
            {tab === "alerts" && <Alerts onSelect={setTicker} />}
          </div>
        )}
      </main>
    </div>
  );
}
