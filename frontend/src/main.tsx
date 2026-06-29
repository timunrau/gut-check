import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

type View = "dump" | "today" | "week" | "patterns" | "logs" | "garmin";
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
  parsed?: Record<string, any> | null;
  event_count?: number;
  events?: EventItem[];
  new_events?: EventItem[];
};

type GarminMetric = {
  metric_date: string;
  steps: number | null;
  sleep_hours: number | null;
  sleep_score: number | null;
  stress_avg: number | null;
  stress_max: number | null;
  body_battery_min: number | null;
  body_battery_max: number | null;
  body_battery_avg: number | null;
  body_battery_end: number | null;
  synced_at: string;
};

type GarminStatus = {
  connected: boolean;
  last_sync_at: string | null;
  last_error: string | null;
  last_success_start_date: string | null;
  last_success_end_date: string | null;
  tokenstore_exists: boolean;
  mfa_pending: boolean;
  auto_sync_enabled: boolean;
  auto_sync_time: string;
  auto_sync_days: number;
  next_auto_sync_at: string | null;
};

type DayResponse = {
  date: string;
  groups: Record<EventType, EventItem[]>;
  garmin: GarminMetric | null;
};

type EventUpdate = Pick<EventItem, "event_date" | "event_time" | "notes">;

type GarminAverages = {
  avg_steps: number | null;
  avg_sleep_hours: number | null;
  avg_sleep_score: number | null;
  avg_stress: number | null;
  avg_body_battery: number | null;
  days_with_data: number;
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
  garmin: {
    days: GarminMetric[];
    averages: GarminAverages;
  };
};

type TriggerCandidate = {
  item: string;
  exposures: number;
  bad_exposures: number;
  tolerated_exposures: number;
  bad_rate: number;
  baseline_bad_rate: number;
  lift: number;
  confidence: "low" | "medium" | "stronger";
  strongest_window: string | null;
  strongest_outcome: string | null;
  language: string;
  evidence: Array<{
    meal_event_id: number;
    outcome_event_id: number;
    meal_at: string;
    outcome_at: string;
    window: string;
    hours_after: number;
    outcomes: string[];
  }>;
};

type PatternsResponse = {
  days: number;
  start_date: string;
  end_date: string;
  counts: {
    meal_exposures: number;
    bad_outcome_events: number;
    baseline_bad_rate: number;
  };
  candidate_triggers: TriggerCandidate[];
  events: EventItem[];
  garmin: {
    days: GarminMetric[];
    averages: GarminAverages;
  };
  summary: string;
  note: string;
};

const bristolTypes = [
  {
    type: 1,
    description: "Separate hard lumps, like little pebbles."
  },
  {
    type: 2,
    description: "Hard and lumpy and starting to resemble a sausage."
  },
  {
    type: 3,
    description: "Sausage-shaped with cracks on the surface."
  },
  {
    type: 4,
    description: "Thinner and more snakelike, plus smooth and soft."
  },
  {
    type: 5,
    description: "Soft blobs with clear-cut edges."
  },
  {
    type: 6,
    description: "Fluffy, mushy pieces with ragged edges."
  },
  {
    type: 7,
    description: "Watery with no solid pieces."
  }
] as const;

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
  if (item.notes?.trim()) return item.notes.trim();

  const data = item.data || {};
  if (item.event_type === "meal") {
    return [...(data.foods || []), ...(data.drinks || []), ...(data.meds || []), ...(data.supplements || [])].join(", ") || item.notes || "Meal";
  }
  if (item.event_type === "bowel_movement") {
    const parts = [
      data.stool_form ? String(data.stool_form) : null,
      data.bristol ? `Bristol ${data.bristol}` : null,
      data.amount ? `${data.amount} amount` : null,
      data.odor ? `${data.odor} odor` : null,
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
    data.context?.note ? String(data.context.note) : null,
    data.stress ? `stress ${data.stress}/5` : null,
    data.sleep_hours ? `sleep ${data.sleep_hours}h` : null,
    ...(data.meds || []),
    ...(data.supplements || [])
  ].filter(Boolean);
  return parts.join(", ") || item.notes || "Context";
}

function compactValue(value: any): string {
  if (value === null || value === undefined || value === "") return "";
  if (Array.isArray(value)) {
    return value.map(compactValue).filter(Boolean).join(", ");
  }
  if (typeof value === "object") {
    return Object.entries(value)
      .map(([key, item]) => {
        const formatted = compactValue(item);
        return formatted ? `${key}: ${formatted}` : "";
      })
      .filter(Boolean)
      .join("; ");
  }
  return String(value);
}

function modelEventLine(event: any): string | null {
  if (!event || typeof event !== "object") return null;
  const eventType = typeof event.type === "string" ? event.type.replace(/_/g, " ") : "event";
  const fields = Object.entries(event)
    .filter(([key]) => !["type", "time", "date_offset", "confidence"].includes(key))
    .map(([key, value]) => {
      const formatted = compactValue(value);
      return formatted ? `${key}: ${formatted}` : "";
    })
    .filter(Boolean);
  return fields.length ? `${eventType}: ${fields.join(" · ")}` : null;
}

function modelEventLines(parsed: Record<string, any> | null | undefined): string[] {
  if (!parsed || !Array.isArray(parsed.events)) return [];
  return parsed.events.map(modelEventLine).filter((line): line is string => Boolean(line));
}

function formatOptionalNumber(value: number | null | undefined, suffix = "", digits = 0): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
  return `${value.toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits })}${suffix}`;
}

function formatOptionalDateTime(value: string | null | undefined): string {
  if (!value) return "Never";
  return new Date(value).toLocaleString();
}

function formatEventTime(value: string): string {
  const match = /^(\d{1,2}):(\d{2})$/.exec(value);
  if (!match) return value;

  const hour = Number(match[1]);
  const minute = match[2];
  if (hour < 0 || hour > 23) return value;

  const period = hour < 12 ? "am" : "pm";
  const displayHour = hour % 12 || 12;
  return `${displayHour}:${minute} ${period}`;
}

function formatBodyBatteryRange(metric: GarminMetric | null | undefined): string {
  if (!metric || metric.body_battery_min === null || metric.body_battery_max === null) return "n/a";
  return `${formatOptionalNumber(metric.body_battery_min)}-${formatOptionalNumber(metric.body_battery_max)}`;
}

function GarminMetricStrip({ metric }: { metric: GarminMetric | null | undefined }) {
  return (
    <section className="wearable-strip" aria-label="Garmin daily metrics">
      <div className="wearable-item">
        <span>{formatOptionalNumber(metric?.steps)}</span>
        <p>Steps</p>
      </div>
      <div className="wearable-item">
        <span>{formatOptionalNumber(metric?.sleep_hours, "h", 1)}</span>
        <p>Sleep</p>
      </div>
      <div className="wearable-item">
        <span>{formatOptionalNumber(metric?.sleep_score)}</span>
        <p>Sleep score</p>
      </div>
      <div className="wearable-item">
        <span>{formatOptionalNumber(metric?.stress_avg, "", 0)}</span>
        <p>Stress avg</p>
      </div>
      <div className="wearable-item">
        <span>{formatBodyBatteryRange(metric)}</span>
        <p>Body battery low/high</p>
      </div>
    </section>
  );
}

async function copyTextToClipboard(text: string): Promise<void> {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }

  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "");
  textarea.style.position = "fixed";
  textarea.style.left = "-9999px";
  textarea.style.top = "0";
  document.body.appendChild(textarea);
  textarea.select();
  const copied = document.execCommand("copy");
  document.body.removeChild(textarea);
  if (!copied) throw new Error("Clipboard copy failed");
}

function cleanForExport(value: any): any {
  if (value === null || value === undefined || value === "") return undefined;
  if (Array.isArray(value)) {
    const items = value.map(cleanForExport).filter((item) => item !== undefined);
    return items.length ? items : undefined;
  }
  if (typeof value === "object") {
    const entries = Object.entries(value)
      .map(([key, item]) => [key, cleanForExport(item)] as const)
      .filter(([, item]) => item !== undefined);
    return entries.length ? Object.fromEntries(entries) : undefined;
  }
  return value;
}

function cleanObjectArrayForExport<T>(items: T[]): any[] {
  return items.map((item) => cleanForExport(item)).filter((item) => item !== undefined);
}

function cleanEventForExport(event: EventItem): Record<string, any> {
  return cleanForExport({
    date: event.event_date,
    time: event.event_time,
    type: event.event_type,
    notes: event.notes,
    data: event.data
  }) || {};
}

function buildPatternsExport(patterns: PatternsResponse): string {
  return JSON.stringify(
    {
      source: "Gut Check",
      export_type: "patterns_period",
      exported_at: new Date().toISOString(),
      selected_period: {
        days: patterns.days,
        start_date: patterns.start_date,
        end_date: patterns.end_date
      },
      pattern_analysis: {
        counts: patterns.counts,
        summary: patterns.summary,
        note: patterns.note,
        candidate_triggers: cleanObjectArrayForExport(patterns.candidate_triggers)
      },
      cleaned_events: patterns.events.map(cleanEventForExport),
      garmin: {
        days: cleanObjectArrayForExport(patterns.garmin.days),
        averages: cleanForExport(patterns.garmin.averages) || {}
      }
    },
    null,
    2
  );
}

function Badge({ children, tone = "neutral" }: { children: React.ReactNode; tone?: "neutral" | "warn" | "good" }) {
  return <span className={`badge ${tone}`}>{children}</span>;
}

function BristolIllustration({ type }: { type: number }) {
  if (type === 1) {
    return (
      <svg className="bristol-illustration" viewBox="0 0 150 88" role="img" aria-label="Bristol type 1 illustration">
        <circle className="bristol-number" cx="18" cy="18" r="14" />
        <text className="bristol-number-text" x="18" y="23">1</text>
        <g fill="#7a473d">
          <ellipse cx="47" cy="26" rx="16" ry="14" transform="rotate(20 47 26)" />
          <ellipse cx="78" cy="31" rx="18" ry="14" transform="rotate(-12 78 31)" />
          <ellipse cx="111" cy="29" rx="17" ry="14" transform="rotate(28 111 29)" />
          <ellipse cx="59" cy="58" rx="18" ry="14" transform="rotate(25 59 58)" />
          <ellipse cx="92" cy="60" rx="16" ry="18" transform="rotate(-24 92 60)" />
          <ellipse cx="125" cy="57" rx="16" ry="15" transform="rotate(35 125 57)" />
        </g>
        <g fill="#9a6356" opacity="0.62">
          <path d="M40 17c9 1 14 7 10 12-7-1-8-6-10-12Z" />
          <path d="M76 22c8 0 14 4 13 11-8 0-11-4-13-11Z" />
          <path d="M52 52c10 1 15 5 15 12-8 0-14-4-15-12Z" />
          <path d="M90 48c8 3 12 8 9 16-8-3-10-8-9-16Z" />
        </g>
      </svg>
    );
  }

  if (type === 2) {
    return (
      <svg className="bristol-illustration" viewBox="0 0 150 88" role="img" aria-label="Bristol type 2 illustration">
        <circle className="bristol-number" cx="18" cy="18" r="14" />
        <text className="bristol-number-text" x="18" y="23">2</text>
        <g fill="#824d43">
          <ellipse cx="53" cy="42" rx="28" ry="24" />
          <ellipse cx="79" cy="39" rx="27" ry="26" />
          <ellipse cx="105" cy="43" rx="29" ry="23" />
          <ellipse cx="126" cy="45" rx="20" ry="18" />
          <ellipse cx="38" cy="46" rx="20" ry="18" />
        </g>
        <g fill="#9d6655" opacity="0.68">
          <path d="M45 27c10 1 16 8 12 14-11-1-11-8-12-14Z" />
          <path d="M71 29c11 2 17 8 15 16-11-2-13-8-15-16Z" />
          <path d="M100 31c14 1 22 8 19 17-13-1-15-8-19-17Z" />
          <path d="M55 50c12 0 18 6 16 14-11 0-14-6-16-14Z" />
          <path d="M112 51c11 0 17 5 16 12-10 1-13-4-16-12Z" />
        </g>
      </svg>
    );
  }

  if (type === 3) {
    return (
      <svg className="bristol-illustration" viewBox="0 0 150 88" role="img" aria-label="Bristol type 3 illustration">
        <circle className="bristol-number" cx="18" cy="18" r="14" />
        <text className="bristol-number-text" x="18" y="23">3</text>
        <path fill="#a36c52" d="M32 39c5-15 24-15 36-14 10 1 17-2 27 0 20 3 34 14 35 27 0 11-7 17-18 18-20 1-27-6-43-4-19 3-33 1-41-7-5-5-4-14 4-20Z" />
        <path fill="#875244" d="M28 55c20 1 33 5 47 3 12-1 19 5 36 2 9-1 15-3 19-6-1 12-10 20-24 20-14 0-20-5-33-3-16 3-36 1-44-8-2-2-2-5-1-8Z" />
        <g fill="#b98468" opacity="0.58">
          <path d="M52 32c12-2 22-1 28 2-6 5-18 6-28-2Z" />
          <path d="M87 30c13 1 23 5 28 12-12 0-22-4-28-12Z" />
        </g>
      </svg>
    );
  }

  if (type === 4) {
    return (
      <svg className="bristol-illustration" viewBox="0 0 150 88" role="img" aria-label="Bristol type 4 illustration">
        <circle className="bristol-number" cx="18" cy="18" r="14" />
        <text className="bristol-number-text" x="18" y="23">4</text>
        <path fill="#a56a4f" d="M29 45c8-15 42-17 72-15 17 1 34-10 42-4 7 6-8 22-34 29-21 6-56-2-82-3-8 0-5-5 2-7Z" />
        <path fill="#bd825e" opacity="0.7" d="M27 49c26-7 58-1 87 3-25 9-67 0-88 1-7 0-6-2 1-4Z" />
      </svg>
    );
  }

  if (type === 5) {
    return (
      <svg className="bristol-illustration" viewBox="0 0 150 88" role="img" aria-label="Bristol type 5 illustration">
        <circle className="bristol-number" cx="18" cy="18" r="14" />
        <text className="bristol-number-text" x="18" y="23">5</text>
        <g fill="#b87856">
          <path d="M31 28c7-11 26-14 35-8 6 4-2 13-15 15-14 3-27 3-20-7Z" />
          <path d="M75 27c10-5 34-3 39 5 4 7-8 12-23 12-17 0-28-9-16-17Z" />
          <path d="M120 26c10-4 24-5 28 1 4 7-8 11-23 12-9 0-14-7-5-13Z" />
          <path d="M28 54c7-10 36-17 51-9 13 7-4 23-28 25-21 2-34-5-23-16Z" />
          <path d="M83 61c4-12 30-22 43-13 12 9 1 24-21 26-15 2-26-2-22-13Z" />
          <path d="M126 52c11-4 33-1 38 8 5 10-16 16-34 14-13-1-18-15-4-22Z" />
        </g>
        <g fill="#d09a73" opacity="0.45">
          <path d="M37 24c8-6 18-8 25-4-7 4-18 7-25 4Z" />
          <path d="M79 30c10-4 21-2 30 3-9 2-22 2-30-3Z" />
          <path d="M33 51c13-5 30-8 42-3-13 8-31 10-42 3Z" />
        </g>
      </svg>
    );
  }

  if (type === 6) {
    return (
      <svg className="bristol-illustration" viewBox="0 0 150 88" role="img" aria-label="Bristol type 6 illustration">
        <circle className="bristol-number" cx="18" cy="18" r="14" />
        <text className="bristol-number-text" x="18" y="23">6</text>
        <g fill="#c88a56">
          <path d="M39 27l10-7 16 1 10-5 15 3 8-3 12 5 10-3 12 8-8 9-21 2-11 6-19-3-11 5-18-3-9 3-7-13 11-5Z" />
          <path d="M30 60l7-9 10 2 10-5 15 5 12-3 11 8-6 12-18 4-10-4-13 5-13-3-5-12Z" />
          <path d="M93 62l17-4 15-2 8 8-5 12-20 4-17 0 2-18Z" />
          <path d="M106 47l15-5 20 3 8 11-25-2-13 4-5-11Z" />
          <path d="M75 49l13-4 13 5-7 10-15 2-4-13Z" />
        </g>
        <g fill="#e1ad75" opacity="0.52">
          <path d="M48 25c17-3 44-3 66 3-19 3-52 3-66-3Z" />
          <path d="M37 58c10-5 24-4 36 2-14 5-28 4-36-2Z" />
          <path d="M108 65c8-4 17-5 24-2-7 5-18 7-24 2Z" />
        </g>
      </svg>
    );
  }

  return (
    <svg className="bristol-illustration" viewBox="0 0 150 88" role="img" aria-label="Bristol type 7 illustration">
      <circle className="bristol-number" cx="18" cy="18" r="14" />
      <text className="bristol-number-text" x="18" y="23">7</text>
      <path fill="#e8ad64" d="M34 41c10-5 21-4 31-10 15-8 26 1 40-4 13-5 34 5 30 17-2 7-16 7-24 13-9 6-13 17-34 16-17-1-25-11-40-11-14 0-21-14-3-21Z" />
      <path fill="#f0c17f" opacity="0.62" d="M55 36c17-9 36-7 42-1-13 7-32 10-42 1Z" />
      <path fill="#f0c17f" opacity="0.62" d="M94 34c16-2 29 1 33 10-12 2-26-1-33-10Z" />
      <path fill="#f0c17f" opacity="0.62" d="M73 54c13-5 28-6 34 2-10 5-26 5-34-2Z" />
      <path fill="#f0c17f" opacity="0.62" d="M47 50c8-1 15 2 15 9-9 1-15-2-15-9Z" />
    </svg>
  );
}

function appendLogPhrase(current: string, phrase: string): string {
  const trimmedEnd = current.trimEnd();
  if (!trimmedEnd) return phrase;
  return `${trimmedEnd}\n${phrase}`;
}

function BristolGuide({ onSelect }: { onSelect: (type: number) => void }) {
  return (
    <details className="bristol-guide">
      <summary>Bristol guide</summary>
      <div className="bristol-panel">
        {bristolTypes.map((item) => (
          <button className="bristol-type" key={item.type} type="button" onClick={() => onSelect(item.type)}>
            <BristolIllustration type={item.type} />
            <div>
              <h2>Type {item.type}</h2>
              <p className="muted">{item.description}</p>
            </div>
          </button>
        ))}
      </div>
    </details>
  );
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

function EventCard({
  item,
  onDelete,
  onUpdate
}: {
  item: EventItem;
  onDelete?: (id: number) => void;
  onUpdate?: (id: number, payload: EventUpdate) => Promise<void>;
}) {
  const [editing, setEditing] = useState(false);
  const [eventDate, setEventDate] = useState(item.event_date);
  const [eventTime, setEventTime] = useState(item.event_time);
  const [notes, setNotes] = useState(() => item.notes || formatEvent(item));
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const summary = formatEvent(item);
  const showNotes = item.notes && item.notes !== summary;

  useEffect(() => {
    if (editing) return;
    setEventDate(item.event_date);
    setEventTime(item.event_time);
    setNotes(item.notes || formatEvent(item));
  }, [editing, item]);

  async function save(event: React.FormEvent) {
    event.preventDefault();
    if (!onUpdate) return;
    setBusy(true);
    setError("");
    try {
      await onUpdate(item.id, {
        event_date: eventDate,
        event_time: eventTime,
        notes: notes.trim() || null
      });
      setEditing(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Update failed");
    } finally {
      setBusy(false);
    }
  }

  if (editing) {
    return (
      <form className="card log-card event-edit-card" onSubmit={save}>
        <div className="event-edit-grid">
          <label>
            <span className="field-label">Date</span>
            <input
              id={`event-${item.id}-date`}
              name={`event-${item.id}-date`}
              type="date"
              value={eventDate}
              onChange={(event) => setEventDate(event.target.value)}
              required
            />
          </label>
          <label>
            <span className="field-label">Time</span>
            <input
              id={`event-${item.id}-time`}
              name={`event-${item.id}-time`}
              type="time"
              value={eventTime}
              onChange={(event) => setEventTime(event.target.value)}
              required
            />
          </label>
        </div>
        <label>
          <span className="field-label">Entry</span>
          <textarea
            className="compact-textarea"
            id={`event-${item.id}-notes`}
            name={`event-${item.id}-notes`}
            value={notes}
            onChange={(event) => setNotes(event.target.value)}
            rows={3}
          />
        </label>
        {error && <p className="error">{error}</p>}
        <div className="action-row">
          <button className="primary small" disabled={busy || !eventDate || !eventTime}>{busy ? "Saving..." : "Save"}</button>
          <button className="ghost small" type="button" disabled={busy} onClick={() => setEditing(false)}>Cancel</button>
        </div>
      </form>
    );
  }

  return (
    <article className="card event-card">
      <div>
        <div className="event-time">
          <strong>{formatEventTime(item.event_time)}</strong>
        </div>
        <p>{summary}</p>
        {showNotes && <p className="muted">{item.notes}</p>}
      </div>
      {(onDelete || onUpdate) && (
        <div className="event-actions">
          {onUpdate && <button className="ghost small" onClick={() => setEditing(true)}>Edit</button>}
          {onDelete && <button className="danger small" onClick={() => onDelete(item.id)}>Delete</button>}
        </div>
      )}
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
        <BristolGuide onSelect={(type) => setDraft((value) => appendLogPhrase(value, `poop bristol ${type}`))} />
        {error && <p className="error">{error}</p>}
      </section>

      {result && (
        <section className="stack">
          <div className="status-line">
            <Badge>{result.entry_classification}</Badge>
            <Badge tone={result.parser_status === "parsed" ? "good" : "warn"}>{result.parser_status}</Badge>
          </div>
          {result.parser_status === "pending" && <p className="warning">Saved raw entry. AI parsing will continue in the background.</p>}
          {result.parser_status === "failed" && <p className="warning">Saved raw entry. The model did not return structured events for this one.</p>}
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

  async function updateEvent(id: number, payload: EventUpdate) {
    await apiFetch<EventItem>(`/api/events/${id}`, {
      method: "PATCH",
      body: JSON.stringify(payload)
    });
    const updatedDate = payload.event_date || dateValue;
    const updated = await apiFetch<DayResponse>(`/api/day/${updatedDate}`);
    setDateValue(updatedDate);
    setDay(updated);
  }

  return (
    <main className="screen">
      <div className="top-row">
        <h1>Today</h1>
        <input id="today-date" name="today-date" type="date" value={dateValue} onChange={(event) => setDateValue(event.target.value)} />
      </div>
      <GarminMetricStrip metric={day?.garmin} />
      {(["meal", "bowel_movement", "symptom", "context"] as EventType[]).map((type) => (
        <section className="stack" key={type}>
          <h2>{labels[type]}</h2>
          {day?.groups[type]?.length ? day.groups[type].map((item) => (
            <EventCard key={item.id} item={item} onDelete={deleteEvent} onUpdate={updateEvent} />
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
        <h2>Garmin averages</h2>
        <div className="metric-grid wearable-grid">
          <div className="metric"><span>{formatOptionalNumber(week?.garmin.averages.avg_steps)}</span><p>Steps</p></div>
          <div className="metric"><span>{formatOptionalNumber(week?.garmin.averages.avg_sleep_hours, "h", 1)}</span><p>Sleep</p></div>
          <div className="metric"><span>{formatOptionalNumber(week?.garmin.averages.avg_sleep_score)}</span><p>Sleep score</p></div>
          <div className="metric"><span>{formatOptionalNumber(week?.garmin.averages.avg_stress, "", 0)}</span><p>Stress</p></div>
          <div className="metric"><span>{formatOptionalNumber(week?.garmin.averages.avg_body_battery, "", 0)}</span><p>Body battery</p></div>
        </div>
        <p className="muted">{week?.garmin.averages.days_with_data ?? 0} days with Garmin data.</p>
        <div className="daily-metric-list">
          {week?.garmin.days.length ? week.garmin.days.map((day) => (
            <article className="daily-metric-row" key={day.metric_date}>
              <strong>{day.metric_date}</strong>
              <span>BB {formatBodyBatteryRange(day)}</span>
              <span>Sleep {formatOptionalNumber(day.sleep_hours, "h", 1)}</span>
              <span>Score {formatOptionalNumber(day.sleep_score)}</span>
            </article>
          )) : <p className="empty">No Garmin days synced.</p>}
        </div>
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

function PatternsView({ refreshToken }: { refreshToken: number }) {
  const [days, setDays] = useState(60);
  const [patterns, setPatterns] = useState<PatternsResponse | null>(null);
  const [copyStatus, setCopyStatus] = useState("");

  useEffect(() => {
    setCopyStatus("");
    apiFetch<PatternsResponse>(`/api/patterns?days=${days}`).then(setPatterns);
  }, [days, refreshToken]);

  async function copyPeriodData() {
    if (!patterns) return;
    setCopyStatus("Copying...");
    try {
      await copyTextToClipboard(buildPatternsExport(patterns));
      setCopyStatus(`Copied ${patterns.events.length} cleaned events and ${patterns.garmin.days.length} Garmin days.`);
    } catch (err) {
      setCopyStatus(err instanceof Error ? err.message : "Could not copy data.");
    }
  }

  return (
    <main className="screen">
      <div className="top-row">
        <h1>Patterns</h1>
        <div className="pattern-actions">
          <select id="pattern-days" name="pattern-days" value={days} onChange={(event) => setDays(Number(event.target.value))}>
            <option value={30}>30 days</option>
            <option value={60}>60 days</option>
            <option value={90}>90 days</option>
            <option value={180}>180 days</option>
          </select>
          <button className="ghost small" type="button" disabled={!patterns || copyStatus === "Copying..."} onClick={copyPeriodData}>
            {copyStatus === "Copying..." ? "Copying..." : "Copy data"}
          </button>
        </div>
      </div>
      {copyStatus && copyStatus !== "Copying..." && <p className={copyStatus.startsWith("Copied") ? "muted" : "error"}>{copyStatus}</p>}
      <section className="stack">
        <article className="card log-card">
          <div className="status-line">
            <Badge>{patterns ? `${patterns.start_date} to ${patterns.end_date}` : "loading"}</Badge>
            <Badge tone="warn">candidate patterns</Badge>
          </div>
          <p>{patterns?.summary || "Checking your logged meals and symptoms..."}</p>
          {patterns?.note && <p className="muted">{patterns.note}</p>}
        </article>
      </section>
      <section className="metric-grid">
        <div className="metric"><span>{patterns?.counts.meal_exposures ?? 0}</span><p>Meal exposures</p></div>
        <div className="metric"><span>{patterns?.counts.bad_outcome_events ?? 0}</span><p>Bad outcomes</p></div>
        <div className="metric"><span>{Math.round((patterns?.counts.baseline_bad_rate ?? 0) * 100)}%</span><p>Baseline bad rate</p></div>
      </section>
      <section className="stack">
        <h2>Candidate triggers</h2>
        {patterns?.candidate_triggers.length ? patterns.candidate_triggers.map((candidate) => (
          <article className="card log-card" key={candidate.item}>
            <div className="status-line">
              <Badge tone={candidate.confidence === "stronger" ? "warn" : "neutral"}>{candidate.confidence}</Badge>
              {candidate.strongest_window && <Badge>{candidate.strongest_window}</Badge>}
              {candidate.strongest_outcome && <Badge>{candidate.strongest_outcome}</Badge>}
            </div>
            <p><strong>{candidate.item}</strong></p>
            <p>{candidate.language}</p>
            <p className="muted">
              {candidate.bad_exposures}/{candidate.exposures} bad after eating; {candidate.tolerated_exposures} tolerated. Baseline {Math.round(candidate.baseline_bad_rate * 100)}%.
            </p>
            {candidate.evidence.length > 0 && (
              <div className="evidence-list">
                {candidate.evidence.map((item) => (
                  <p className="muted" key={`${candidate.item}-${item.meal_event_id}-${item.outcome_event_id}`}>
                    {new Date(item.meal_at).toLocaleString()} led to {item.outcomes.join(", ")} {item.hours_after}h later
                  </p>
                ))}
              </div>
            )}
          </article>
        )) : <p className="empty">No candidate patterns yet.</p>}
      </section>
    </main>
  );
}

function GarminView({ refreshKey }: { refreshKey: () => void }) {
  const [status, setStatus] = useState<GarminStatus | null>(null);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [pendingId, setPendingId] = useState<string | null>(null);
  const [mfaCode, setMfaCode] = useState("");
  const [busy, setBusy] = useState("");
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  async function loadStatus() {
    setStatus(await apiFetch<GarminStatus>("/api/garmin/status"));
  }

  useEffect(() => {
    loadStatus().catch((err) => setError(err instanceof Error ? err.message : "Could not load Garmin status"));
  }, []);

  async function startAuth(event: React.FormEvent) {
    event.preventDefault();
    setBusy("login");
    setError("");
    setMessage("");
    try {
      const result = await apiFetch<{ connected: boolean; mfa_required: boolean; pending_id: string | null }>("/api/garmin/auth/start", {
        method: "POST",
        body: JSON.stringify({ email, password })
      });
      setPassword("");
      if (result.mfa_required && result.pending_id) {
        setPendingId(result.pending_id);
        setMessage("Enter your Garmin MFA code to finish connecting.");
      } else {
        setPendingId(null);
        setMessage("Garmin connected.");
      }
      await loadStatus();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Garmin login failed");
    } finally {
      setBusy("");
    }
  }

  async function finishAuth(event: React.FormEvent) {
    event.preventDefault();
    if (!pendingId) return;
    setBusy("mfa");
    setError("");
    setMessage("");
    try {
      await apiFetch("/api/garmin/auth/finish", {
        method: "POST",
        body: JSON.stringify({ pending_id: pendingId, mfa_code: mfaCode })
      });
      setPendingId(null);
      setMfaCode("");
      setMessage("Garmin connected.");
      await loadStatus();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Garmin MFA failed");
    } finally {
      setBusy("");
    }
  }

  async function testConnection() {
    setBusy("test");
    setError("");
    setMessage("");
    try {
      const result = await apiFetch<{ ok: boolean; date: string; steps: number | null }>("/api/garmin/test", { method: "POST" });
      setMessage(`Connection works for ${result.date}; steps ${formatOptionalNumber(result.steps)}.`);
      await loadStatus();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Garmin test failed");
    } finally {
      setBusy("");
    }
  }

  async function sync() {
    setBusy("sync");
    setError("");
    setMessage("");
    try {
      const result = await apiFetch<{ synced: number; start_date: string; end_date: string }>("/api/garmin/sync", {
        method: "POST",
        body: JSON.stringify({ days: 14 })
      });
      setMessage(`Synced ${result.synced} days from ${result.start_date} to ${result.end_date}.`);
      refreshKey();
      await loadStatus();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Garmin sync failed");
    } finally {
      setBusy("");
    }
  }

  return (
    <main className="screen">
      <div className="top-row">
        <h1>Garmin</h1>
        <Badge tone={status?.connected ? "good" : "warn"}>{status?.connected ? "connected" : "not connected"}</Badge>
      </div>

      <section className="card log-card">
        <div className="status-line">
          <Badge>{status?.tokenstore_exists ? "tokens saved" : "no tokens"}</Badge>
          {status?.mfa_pending && <Badge tone="warn">MFA pending</Badge>}
        </div>
        <p className="muted">Last sync: {formatOptionalDateTime(status?.last_sync_at)}</p>
        {status?.last_success_start_date && status.last_success_end_date && (
          <p className="muted">Range: {status.last_success_start_date} to {status.last_success_end_date}</p>
        )}
        <p className="muted">
          Nightly sync: {status?.auto_sync_enabled ? `${status.auto_sync_days} days at ${status.auto_sync_time}` : "off"}
        </p>
        {status?.next_auto_sync_at && <p className="muted">Next auto sync: {formatOptionalDateTime(status.next_auto_sync_at)}</p>}
        {status?.last_error && <p className="warning">{status.last_error}</p>}
        <div className="action-row">
          <button className="ghost" onClick={testConnection} disabled={Boolean(busy) || !status?.tokenstore_exists}>
            {busy === "test" ? "Testing..." : "Test"}
          </button>
          <button className="primary" onClick={sync} disabled={Boolean(busy) || !status?.tokenstore_exists}>
            {busy === "sync" ? "Syncing..." : "Sync 14 days"}
          </button>
          <button className="ghost" onClick={loadStatus} disabled={Boolean(busy)}>Refresh</button>
        </div>
      </section>

      <form className="card log-card" onSubmit={startAuth}>
        <h2>Connect</h2>
        <input
          id="garmin-email"
          name="garmin-email"
          type="email"
          autoComplete="username"
          value={email}
          onChange={(event) => setEmail(event.target.value)}
          placeholder="Garmin email"
          aria-label="Garmin email"
        />
        <input
          id="garmin-password"
          name="garmin-password"
          type="password"
          autoComplete="current-password"
          value={password}
          onChange={(event) => setPassword(event.target.value)}
          placeholder="Garmin password"
          aria-label="Garmin password"
        />
        <button className="primary" disabled={busy === "login" || !email.trim() || !password}>
          {busy === "login" ? "Connecting..." : "Connect Garmin"}
        </button>
      </form>

      {pendingId && (
        <form className="card log-card" onSubmit={finishAuth}>
          <h2>MFA</h2>
          <input
            id="garmin-mfa"
            name="garmin-mfa"
            type="text"
            inputMode="numeric"
            autoComplete="one-time-code"
            value={mfaCode}
            onChange={(event) => setMfaCode(event.target.value)}
            placeholder="MFA code"
            aria-label="Garmin MFA code"
          />
          <button className="primary" disabled={busy === "mfa" || !mfaCode.trim()}>
            {busy === "mfa" ? "Verifying..." : "Finish connection"}
          </button>
        </form>
      )}

      {message && <p className="muted">{message}</p>}
      {error && <p className="error">{error}</p>}
    </main>
  );
}

function LogCard({
  log,
  busyId,
  onReparse,
  onRemove
}: {
  log: LogItem;
  busyId: number | null;
  onReparse: (id: number) => void;
  onRemove: (id: number) => void;
}) {
  const lines = modelEventLines(log.parsed);
  const summary = typeof log.parsed?.summary === "string" ? log.parsed.summary : "";

  return (
    <article className="card log-card">
      <div className="status-line">
        <Badge>{log.entry_classification}</Badge>
        <Badge tone={log.parser_status === "parsed" ? "good" : "warn"}>{log.parser_status}</Badge>
      </div>
      <p>{log.raw_text}</p>
      {(summary || lines.length > 0) && (
        <div className="ai-panel">
          {summary && <p>{summary}</p>}
          {lines.length > 0 && (
            <div className="ai-lines">
              {lines.map((line, index) => <p className="muted" key={`${log.id}-${index}`}>{line}</p>)}
            </div>
          )}
        </div>
      )}
      {log.parser_error && <p className="warning">Parser: {log.parser_error}</p>}
      <p className="muted">{new Date(log.created_at).toLocaleString()} · {log.event_count ?? 0} events</p>
      <div className="action-row">
        <button className="ghost" disabled={busyId === log.id} onClick={() => onReparse(log.id)}>Reparse</button>
        <button className="danger" disabled={busyId === log.id} onClick={() => onRemove(log.id)}>Delete</button>
      </div>
    </article>
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
          <LogCard key={log.id} log={log} busyId={busyId} onReparse={reparse} onRemove={remove} />
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
    if (view === "patterns") return <PatternsView refreshToken={refreshToken} />;
    if (view === "logs") return <LogsView refreshKey={refreshKey} />;
    if (view === "garmin") return <GarminView refreshKey={refreshKey} />;
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
        {(["dump", "today", "week", "patterns", "logs", "garmin"] as View[]).map((item) => (
          <button key={item} className={view === item ? "active" : ""} onClick={() => setView(item)}>
            {item[0].toUpperCase() + item.slice(1)}
          </button>
        ))}
      </nav>
    </>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
