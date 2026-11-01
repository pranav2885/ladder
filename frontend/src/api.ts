// Thin typed wrappers over the Ladder API. Every number shown in the UI comes from here.

export type Summary = {
  n: number;
  accuracy: number;
  joules_total: number;
  joules_mean?: number;
  seconds_mean?: number;
  seconds_p95?: number;
  final_tier_share?: number[];
  escalation_rate?: number[];
  accepted_wrong_rate?: number;
  wasted_joules_total?: number;
  inr?: number;
  g_co2?: number;
};

export type ModePoint = {
  test: Summary;
  thresholds?: number[];
  preroute_thresholds?: number[];
  saved_vs_largest_pct?: number;
  oracle_savings_achieved_pct?: number;
};

export type LadderPoint = ModePoint & {
  target: number;
  dev_accuracy?: number;
  random_same_escalation?: Summary;
  preroute?: ModePoint;
  hybrid?: ModePoint;
};

export type TaskResult = {
  task: string;
  tiers?: string[];
  models?: string[];
  n_test?: number;
  energy_kinds?: string[];
  baselines: Record<string, Summary>;
  ladder: LadderPoint[];
};

export type Results = {
  created_at: string;
  targets: number[];
  tariff: { inr_per_kwh: number; grid_kg_co2_per_kwh: number };
  tasks: TaskResult[];
  missing_tasks: string[];
  mixed: TaskResult;
};

export type Step = {
  tier: string;
  model: string;
  text: string;
  confidence: number | null;
  threshold: number | null;
  accepted: boolean;
  joules: number;
  seconds: number;
  tokens: number;
  energy_kind: string;
  correct?: boolean;
};

export type Trace = {
  task: string;
  quality: string;
  steps: Step[];
  final_tier: string;
  answer: string;
  correct: boolean | null;
  joules_total: number;
  seconds_total: number;
  wasted_joules: number;
  inr: number;
  g_co2: number;
  largest_joules_estimate: number;
  saved_joules_estimate: number;
  energy_kind: string;
  notes: string;
};

export type TaskInfo = Record<
  string,
  {
    tiers: { name: string; model: string }[];
    thresholds: Record<string, number[]>;
    input_fields: string[];
    largest_joules_estimate: number;
  }
>;

async function get<T>(path: string): Promise<T | null> {
  const r = await fetch(path);
  if (r.status === 404) return null;
  if (!r.ok) throw new Error(`${path}: ${r.status} ${await r.text()}`);
  return r.json();
}

export const api = {
  health: () => get<{ measurement_in_progress: boolean; router_loaded: boolean }>("/api/health"),
  results: () => get<Results>("/api/results"),
  machine: () => get<any>("/api/machine"),
  tasks: () => get<TaskInfo>("/api/tasks"),
  example: (task: string) => get<{ input: Record<string, string>; gold: unknown }>(`/api/example?task=${task}`),
  ladder: async (body: { task: string; input: Record<string, string>; quality: string; gold?: unknown }) => {
    const r = await fetch("/api/ladder", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) throw new Error(await r.text());
    return (await r.json()) as Trace;
  },
};

export const MODES = [
  { key: "cascade", label: "Cascade", color: "var(--series-1)" },
  { key: "preroute", label: "Pre-routing", color: "var(--series-2)" },
  { key: "hybrid", label: "Hybrid", color: "var(--series-3)" },
] as const;
export type ModeKey = (typeof MODES)[number]["key"];

export function modePoint(p: LadderPoint, mode: ModeKey): ModePoint | undefined {
  return mode === "cascade" ? p : p[mode];
}
