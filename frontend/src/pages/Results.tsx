import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  LabelList,
  Legend,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, MODES, modePoint, type Results as R, type TaskResult } from "../api";
import { Card, Chips, Empty, Stat, Table, num, pct, signedPct } from "../ui";

const AXIS = { stroke: "var(--axis)", tick: { fill: "var(--muted)", fontSize: 12 } };
const TIER_COLORS = ["var(--tier-1)", "var(--tier-2)", "var(--tier-3)"];

type Pt = { x: number; y: number; name: string; target?: number };

function TipBox({ active, payload }: any) {
  if (!active || !payload?.length) return null;
  const p: Pt = payload[0].payload;
  return (
    <div className="rounded-lg border border-line bg-surface px-3 py-2 text-xs shadow-sm">
      <div className="font-medium text-ink">{p.name}</div>
      {p.target !== undefined && <div className="text-ink2">quality target: within {pct(p.target, 0)}</div>}
      <div className="num text-ink">accuracy {pct(p.y)}</div>
      <div className="num text-ink">{num(p.x)} J per request</div>
    </div>
  );
}

export function useResults() {
  const [data, setData] = useState<R | null | undefined>(undefined);
  const [error, setError] = useState<string>();
  useEffect(() => {
    api.results().then(setData).catch((e) => setError(String(e)));
  }, []);
  return { data, error };
}

export function ResultsGate({ data, error, children }: { data: R | null | undefined; error?: string; children: (d: R) => JSX.Element }) {
  if (error) return <Empty title="Could not load results" hint={error} />;
  if (data === undefined) return <div className="text-sm text-ink2">Loading…</div>;
  if (data === null)
    return <Empty title="No results yet" hint={<>Run <code>ladder cache run</code>, then <code>ladder eval</code>. Nothing is shown until real runs exist.</>} />;
  return children(data);
}

function AccuracyEnergy({ t }: { t: TaskResult }) {
  const n = t.ladder[0]?.test.n || 1;
  const baselines: Pt[] = Object.entries(t.baselines)
    .filter(([k]) => k.startsWith("always_t"))
    .map(([k, b]) => ({ x: b.joules_total / n, y: b.accuracy, name: `always ${k.slice(7)}` }));
  const oracle = t.baselines.oracle;
  const series = MODES.map((m) => ({
    ...m,
    pts: t.ladder
      .map((p) => ({ p, q: modePoint(p, m.key) }))
      .filter((r) => r.q)
      .map(({ p, q }) => ({ x: q!.test.joules_total / n, y: q!.test.accuracy, name: m.label, target: p.target }))
      .sort((a, b) => a.x - b.x),
  })).filter((s) => s.pts.length);
  const random: Pt[] = t.ladder
    .filter((p) => p.random_same_escalation)
    .map((p) => ({ x: p.random_same_escalation!.joules_total / n, y: p.random_same_escalation!.accuracy,
                   name: "Random escalation (same rate as cascade)", target: p.target }));

  const ys = [...baselines, ...random, ...series.flatMap((s) => s.pts)].map((p) => p.y)
    .concat(oracle ? [oracle.accuracy] : []);
  const lo = Math.floor(Math.min(...ys) * 20) / 20;
  const hi = Math.min(1, Math.ceil(Math.max(...ys) * 20) / 20);
  const ticks = Array.from({ length: Math.round((hi - lo) * 20) + 1 }, (_, i) => +(lo + i * 0.05).toFixed(2));

  return (
    <ResponsiveContainer width="100%" height={340}>
      <ScatterChart margin={{ top: 10, right: 24, bottom: 28, left: 4 }}>
        <CartesianGrid stroke="var(--line)" strokeDasharray="" vertical={false} />
        <XAxis type="number" dataKey="x" name="J/request" {...AXIS} domain={["auto", "auto"]}
               label={{ value: "energy per request (J, CPU package, net of idle)", position: "insideBottom", offset: -16, fill: "var(--muted)", fontSize: 12 }} />
        <YAxis type="number" dataKey="y" {...AXIS} domain={[lo, hi]} ticks={ticks} tickFormatter={(v) => `${Math.round(v * 100)}%`} width={44} />
        <Tooltip content={<TipBox />} cursor={{ stroke: "var(--axis)" }} />
        <Legend verticalAlign="top" align="left" height={52} wrapperStyle={{ fontSize: 12 }}
                formatter={(v) => <span style={{ color: "var(--ink2)" }}>{v}</span>} />
        <Scatter name="Baselines (always one tier)" data={baselines} fill="var(--muted)">
          <LabelList dataKey="name" position="right" style={{ fill: "var(--ink2)", fontSize: 11 }} />
        </Scatter>
        {oracle && (
          <Scatter name="Oracle (ceiling)" data={[{ x: oracle.joules_total / n, y: oracle.accuracy, name: "Oracle: cheapest correct tier" }]}
                   fill="var(--ink)" shape="star" />
        )}
        {random.length > 0 && <Scatter name="Random escalation" data={random} fill="var(--ink2)" shape="cross" />}
        {series.map((s) => (
          <Scatter key={s.key} name={`Ladder ${s.label}`} data={s.pts} fill={s.color}
                   line={{ stroke: s.color, strokeWidth: 2 }} lineJointType="linear"
                   stroke="var(--surface)" strokeWidth={2} />
        ))}
      </ScatterChart>
    </ResponsiveContainer>
  );
}

function TierShare({ t, k }: { t: TaskResult; k: number }) {
  const tiers = t.tiers ?? [];
  const rows = MODES.flatMap((m) => {
    const q = modePoint(t.ladder[k], m.key);
    if (!q?.test.final_tier_share) return [];
    return [{ mode: m.label, ...Object.fromEntries(q.test.final_tier_share.map((v, i) => [tiers[i], v])) }];
  });
  if (!rows.length || tiers.length < 2) return <div className="text-sm text-ink2">Only one tier has complete data for this task.</div>;
  return (
    <ResponsiveContainer width="100%" height={60 + rows.length * 36}>
      <BarChart data={rows} layout="vertical" margin={{ top: 0, right: 16, bottom: 0, left: 8 }}>
        <XAxis type="number" domain={[0, 1]} {...AXIS} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
        <YAxis type="category" dataKey="mode" {...AXIS} width={84} />
        <Tooltip formatter={(v: number) => pct(v)} contentStyle={{ background: "var(--surface)", border: "1px solid var(--line)", fontSize: 12 }} />
        <Legend wrapperStyle={{ fontSize: 12 }} formatter={(v) => <span style={{ color: "var(--ink2)" }}>answered by {v}</span>} />
        {tiers.map((tier, i) => (
          <Bar key={tier} dataKey={tier} stackId="a" fill={TIER_COLORS[i]} stroke="var(--surface)" strokeWidth={2}
               radius={i === tiers.length - 1 ? [0, 4, 4, 0] : 0} barSize={18} />
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}

function ResultsBody({ data }: { data: R }) {
  const options = [...data.tasks.map((t) => ({ value: t.task, label: t.task })), { value: "mixed", label: "mixed traffic" }];
  const [task, setTask] = useState(options[0]?.value ?? "mixed");
  const k = Math.max(0, data.targets.indexOf(0.01));
  const t: TaskResult = task === "mixed" ? data.mixed : data.tasks.find((x) => x.task === task)!;
  const big = t.baselines.always_largest;
  // The headline mode is chosen on dev (lowest dev energy that met the target), never on test.
  const best = useMemo(() => {
    const cands = MODES.map((m) => ({ m, q: modePoint(t.ladder[k], m.key) as (ReturnType<typeof modePoint> & { dev_joules?: number }) }))
      .filter((c) => c.q && c.q.dev_joules !== undefined);
    return cands.sort((a, b) => a.q!.dev_joules! - b.q!.dev_joules!)[0]
      ?? MODES.map((m) => ({ m, q: modePoint(t.ladder[k], m.key) })).find((c) => c.q);
  }, [t, k]);
  const saved = best && big ? 1 - best.q!.test.joules_total / big.joules_total : undefined;

  return (
    <div className="space-y-4">
      <Chips label="Task" value={task} options={options} onChange={setTask} />
      {data.missing_tasks.length > 0 && (
        <div className="text-xs text-ink2">Waiting for cache coverage: {data.missing_tasks.join(", ")}.</div>
      )}
      {t.models && <div className="text-xs text-ink2">Ladder for this task: {t.models.join(" → ")} (largest tier with complete data)</div>}
      {big && best && (
        <div className="grid gap-3 sm:grid-cols-3">
          <Stat label="Always-largest accuracy" value={pct(big.accuracy)} sub={`${num(big.joules_total / big.n)} J / request`} />
          <Stat label={`Ladder ${best.m.label} accuracy (target 1%)`} value={pct(best.q!.test.accuracy)}
                sub={`${signedPct(100 * (best.q!.test.accuracy - big.accuracy))} pts vs always-largest`} />
          <Stat label="Energy vs always-largest" value={saved! >= 0 ? `${pct(saved, 0)} less` : `${pct(-saved!, 0)} more`}
                sub={`${best.m.label}: the mode with the lowest dev-split energy at the 1% target; numbers are test split`} />
        </div>
      )}
      <Card title="Accuracy vs energy (test split)">
        <AccuracyEnergy t={t} />
        <p className="mt-2 text-xs text-ink2">Each Ladder point is a quality target (0–10% below always-largest on dev), chosen on the dev split only. Up and to the left is better.</p>
      </Card>
      <Card title="Modes at the 1% quality target">
        <Table
          head={["mode", "accuracy", "J / request", "energy vs largest", "mean s", "p95 s", "escalation per step"]}
          rows={[
            ...Object.entries(t.baselines).map(([name, b]) => [
              name.replace("_", " "), pct(b.accuracy), num(b.joules_total / b.n),
              big ? signedPct(100 * (b.joules_total / big.joules_total - 1)) : "–",
              num(b.seconds_mean, 2), num(b.seconds_p95, 2), "–",
            ]),
            ...MODES.flatMap((m) => {
              const q = modePoint(t.ladder[k], m.key);
              if (!q) return [];
              return [[<span style={{ color: "var(--ink)" }}>Ladder {m.label}</span>, pct(q.test.accuracy), num(q.test.joules_total / q.test.n),
                big ? signedPct(100 * (q.test.joules_total / big.joules_total - 1)) : "–",
                num(q.test.seconds_mean, 2), num(q.test.seconds_p95, 2),
                q.test.escalation_rate?.map((x) => pct(x, 0)).join(", ") || "–"]];
            }),
          ]}
        />
      </Card>
      {task !== "mixed" && (
        <Card title="Which tier answered (1% target)">
          <TierShare t={t} k={k} />
        </Card>
      )}
      <div className="text-xs text-muted">Generated {new Date(data.created_at).toLocaleString()} · energy kinds: {(t.energy_kinds ?? ["see tasks"]).join(", ")}</div>
    </div>
  );
}

export default function Results() {
  const { data, error } = useResults();
  return <ResultsGate data={data} error={error}>{(d) => <ResultsBody data={d} />}</ResultsGate>;
}
