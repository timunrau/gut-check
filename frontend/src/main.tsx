import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

type View = "dump" | "today" | "week" | "logs";
type EventType = "meal" | "bowel_movement" | "symptom" | "context";

type EventItem = {
  id: number;
  raw_log_id: number;
  event_type: EventType;
  event_date: string;
  event_time: string;
  time_was_defaulted: boolean;
  notes: string | null;
  confidence: number;
  data: Record<string, any>;
};

type LogItem = {
  id: number;
  raw_text: string;
  created_at: string;
  parser_status: string;
  parser_error: string | null;
  entry_classification: string;
  classification_confidence: number;
  event_count?: number;
  events?: EventItem[];
  new_events?: EventItem[];
};

type DayResponse = {
  date: string;
  groups: Record<EventType, EventItem[]>;
};

type WeekResponse = {
  start_date: string;
  end_date: string;
  counts: {
    bowel_movements: number;
    high_symptom_bowel_movements: number;
    symptom_entries: number;
  };
  possible_repeated_foods_or_drinks: Array<{ item: string; count: number; language: string }>;
  note: string;
};

const labels: Record<EventType, string> = {
  meal: "Meals",
  bowel_movement: "Bowel movements",
  symptom: "Symptoms",
  context: "Context"
};

function dateInputValue(date: Date): string {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function todayIso(): string {
  return dateInputValue(new Date());
}

function weekStartIso(): string {
  const current = new Date();
  const day = current.getDay();
  const diff = day === 0 ? -6 : 1 - day;
  current.setDate(current.getDate() + diff);
  return dateInputValue(current);
}

async function apiFetch<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  if (init.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const response = await fetch(path, {
    ...init,
    headers,
    credentials: "include"
  });
  if (!response.ok) {
    let message = response.statusText;
    try {
      const data = await response.json();
      message = data.detail || message;
    } catch {
      // Keep the HTTP status text.
    }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

function formatEvent(item: EventItem): string {
  const data = item.data || {};
  if (item.event_type === "meal") {
    return [...(data.foods || []), ...(data.drinks || []), ...(data.meds || []), ...(data.supplements || [])].join(", ") || item.notes || "Meal";
  }
  if (item.event_type === "bowel_movement") {
    const parts = [
      data.bristol ? `Bristol ${data.bristol}` : null,
      data.urgency ? `urgency ${data.urgency}` : null,
      data.pain ? `pain ${data.pain}` : null,
      data.bloating ? `bloating ${data.bloating}` : null
    ].filter(Boolean);
    return parts.join(", ") || item.notes || "Bowel movement";
  }
  if (item.event_type === "symptom") {
    const symptoms = (data.symptoms || []).map((symptom: any) => symptom.severity ? `${symptom.name} ${symptom.severity}/5` : symptom.name);
    return symptoms.join(", ") || item.notes || "Symptom";
  }
  const parts = [
    data.stress ? `stress ${data.stress}/5` : null,
    data.sleep_hours ? `sleep ${data.sleep_hours}h` : null,
    ...(data.meds || []),
    ...(data.supplements || [])
  ].filter(Boolean);
  return parts.join(", ") || item.notes || "Context";
}

function Badge({ children, tone = "neutral" }: { children: React.ReactNode; tone?: "neutral" | "warn" | "good" }) {
  return <span className={`badge ${tone}`}>{children}</span>;
}

function Login({ onLogin }: { onLogin: () => void }) {
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError("");
    try {
      await apiFetch("/api/auth/login", {
        method: "POST",
        body: JSON.stringify({ password })
      });
      onLogin();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Login failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="login-shell">
      <form className="login-panel" onSubmit={submit}>
        <h1>Gut Check</h1>
        <input
          autoFocus
          id="login-password"
          name="password"
          type="password"
          autoComplete="current-password"
          value={password}
          onChange={(event) => setPassword(event.target.value)}
          placeholder="Password"
          aria-label="Password"
        />
        {error && <p className="error">{error}</p>}
        <button className="primary" disabled={busy || !password}>{busy ? "Signing in..." : "Log in"}</button>
      </form>
    </main>
  );
}

function EventCard({ item, onDelete }: { item: EventItem; onDelete?: (id: number) => void }) {
  return (
    <article className="card event-card">
      <div>
        <div className="event-time">
          <strong>{item.event_time}</strong>
          {item.time_was_defaulted && <Badge tone="warn">defaulted</Badge>}
        </div>
        <p>{formatEvent(item)}</p>
        {item.notes && <p className="muted">{item.notes}</p>}
      </div>
      {onDelete && <button className="danger small" onClick={() => onDelete(item.id)}>Delete</button>}
    </article>
  );
}

function DumpView({ refreshKey }: { refreshKey: () => void }) {
  const [draft, setDraft] = useState(() => localStorage.getItem("gutcheck.draft") || "");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<LogItem | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    localStorage.setItem("gutcheck.draft", draft);
  }, [draft]);

  async function save() {
    const rawText = draft.trim();
    if (!rawText) return;
    setBusy(true);
    setError("");
    try {
      const saved = await apiFetch<LogItem>("/api/logs", {
        method: "POST",
        body: JSON.stringify({ raw_text: rawText })
      });
      setResult(saved);
      setDraft("");
      localStorage.removeItem("gutcheck.draft");
      refreshKey();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Save failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="screen">
      <section className="stack">
        <textarea
          id="raw-entry"
          name="raw-entry"
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          placeholder="Say what you ate, pooped, felt, slept, or took..."
          rows={8}
        />
        <div className="action-row">
          <button className="primary" onClick={save} disabled={busy || !draft.trim()}>{busy ? "Saving..." : "Save"}</button>
          <button className="ghost" onClick={() => setDraft("")}>Clear</button>
        </div>
        {error && <p className="error">{error}</p>}
      </section>

      {result && (
        <section className="stack">
          <div className="status-line">
            <Badge>{result.entry_classification}</Badge>
            <Badge tone={result.parser_status === "parsed" ? "good" : "warn"}>{result.parser_status}</Badge>
          </div>
          {result.parser_status !== "parsed" && <p className="warning">Saved raw entry. The model did not return structured events for this one.</p>}
          {(result.events || result.new_events || []).map((item) => <EventCard key={item.id} item={item} />)}
        </section>
      )}
    </main>
  );
}

function TodayView({ refreshToken }: { refreshToken: number }) {
  const [dateValue, setDateValue] = useState(todayIso());
  const [day, setDay] = useState<DayResponse | null>(null);

  useEffect(() => {
    apiFetch<DayResponse>(`/api/day/${dateValue}`).then(setDay);
  }, [dateValue, refreshToken]);

  async function deleteEvent(id: number) {
    await apiFetch(`/api/events/${id}`, { method: "DELETE" });
    const updated = await apiFetch<DayResponse>(`/api/day/${dateValue}`);
    setDay(updated);
  }

  return (
    <main className="screen">
      <div className="top-row">
        <h1>Today</h1>
        <input id="today-date" name="today-date" type="date" value={dateValue} onChange={(event) => setDateValue(event.target.value)} />
      </div>
      {(["meal", "bowel_movement", "symptom", "context"] as EventType[]).map((type) => (
        <section className="stack" key={type}>
          <h2>{labels[type]}</h2>
          {day?.groups[type]?.length ? day.groups[type].map((item) => (
            <EventCard key={item.id} item={item} onDelete={deleteEvent} />
          )) : <p className="empty">No entries.</p>}
        </section>
      ))}
    </main>
  );
}

function WeekView({ refreshToken }: { refreshToken: number }) {
  const [startDate, setStartDate] = useState(weekStartIso());
  const [week, setWeek] = useState<WeekResponse | null>(null);

  useEffect(() => {
    apiFetch<WeekResponse>(`/api/week/${startDate}`).then(setWeek);
  }, [startDate, refreshToken]);

  return (
    <main className="screen">
      <div className="top-row">
        <h1>Week</h1>
        <input id="week-start-date" name="week-start-date" type="date" value={startDate} onChange={(event) => setStartDate(event.target.value)} />
      </div>
      <section className="metric-grid">
        <div className="metric"><span>{week?.counts.bowel_movements ?? 0}</span><p>Bowel movements</p></div>
        <div className="metric"><span>{week?.counts.high_symptom_bowel_movements ?? 0}</span><p>High-symptom BMs</p></div>
        <div className="metric"><span>{week?.counts.symptom_entries ?? 0}</span><p>Symptom entries</p></div>
      </section>
      <section className="stack">
        <h2>Worth watching</h2>
        {week?.possible_repeated_foods_or_drinks.length ? week.possible_repeated_foods_or_drinks.map((item) => (
          <article className="card" key={item.item}>
            <p><strong>{item.item}</strong> appeared before {item.count} bad episodes.</p>
            <p className="muted">possible; not confirmed</p>
          </article>
        )) : <p className="empty">insufficient data</p>}
        {week?.note && <p className="muted">{week.note}</p>}
      </section>
    </main>
  );
}

function LogsView({ refreshKey }: { refreshKey: () => void }) {
  const [logs, setLogs] = useState<LogItem[]>([]);
  const [busyId, setBusyId] = useState<number | null>(null);

  async function load() {
    setLogs(await apiFetch<LogItem[]>("/api/logs/recent"));
  }

  useEffect(() => {
    load();
  }, []);

  async function reparse(id: number) {
    setBusyId(id);
    await apiFetch(`/api/logs/${id}/reparse`, { method: "POST" });
    await load();
    refreshKey();
    setBusyId(null);
  }

  async function remove(id: number) {
    setBusyId(id);
    await apiFetch(`/api/logs/${id}`, { method: "DELETE" });
    await load();
    refreshKey();
    setBusyId(null);
  }

  return (
    <main className="screen">
      <div className="top-row">
        <h1>Logs</h1>
        <button className="ghost small" onClick={load}>Refresh</button>
      </div>
      <section className="stack">
        {logs.length ? logs.map((log) => (
          <article className="card log-card" key={log.id}>
            <div className="status-line">
              <Badge>{log.entry_classification}</Badge>
              <Badge tone={log.parser_status === "parsed" ? "good" : "warn"}>{log.parser_status}</Badge>
            </div>
            <p>{log.raw_text}</p>
            {log.parser_error && <p className="warning">Parser: {log.parser_error}</p>}
            <p className="muted">{new Date(log.created_at).toLocaleString()} · {log.event_count ?? 0} events</p>
            <div className="action-row">
              <button className="ghost" disabled={busyId === log.id} onClick={() => reparse(log.id)}>Reparse</button>
              <button className="danger" disabled={busyId === log.id} onClick={() => remove(log.id)}>Delete</button>
            </div>
          </article>
        )) : <p className="empty">No logs.</p>}
      </section>
    </main>
  );
}

function App() {
  const [checking, setChecking] = useState(true);
  const [authed, setAuthed] = useState(false);
  const [view, setView] = useState<View>("dump");
  const [refreshToken, setRefreshToken] = useState(0);

  const refreshKey = () => setRefreshToken((value) => value + 1);

  useEffect(() => {
    apiFetch("/api/auth/me")
      .then(() => setAuthed(true))
      .catch(() => setAuthed(false))
      .finally(() => setChecking(false));
  }, []);

  useEffect(() => {
    if ("serviceWorker" in navigator) {
      navigator.serviceWorker.register("/sw.js").catch(() => undefined);
    }
  }, []);

  const screen = useMemo(() => {
    if (view === "today") return <TodayView refreshToken={refreshToken} />;
    if (view === "week") return <WeekView refreshToken={refreshToken} />;
    if (view === "logs") return <LogsView refreshKey={refreshKey} />;
    return <DumpView refreshKey={refreshKey} />;
  }, [view, refreshToken]);

  if (checking) {
    return <main className="login-shell"><p>Loading...</p></main>;
  }
  if (!authed) {
    return <Login onLogin={() => setAuthed(true)} />;
  }

  return (
    <>
      <header className="app-header">
        <strong>Gut Check</strong>
        <button className="ghost small" onClick={async () => {
          await apiFetch("/api/auth/logout", { method: "POST" });
          setAuthed(false);
        }}>Log out</button>
      </header>
      {screen}
      <nav className="bottom-nav" aria-label="Primary">
        {(["dump", "today", "week", "logs"] as View[]).map((item) => (
          <button key={item} className={view === item ? "active" : ""} onClick={() => setView(item)}>
            {item[0].toUpperCase() + item.slice(1)}
          </button>
        ))}
      </nav>
    </>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
