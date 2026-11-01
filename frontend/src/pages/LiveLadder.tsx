import { useEffect, useState } from "react";
import { api, type OpenRef, type TaskInfo, type Trace } from "../api";
import OpenModels from "../OpenModels";
import { Card, Chips, Empty, num } from "../ui";

const QUALITIES = [
  { value: "fast", label: "Fast (within 10%)" },
  { value: "balanced", label: "Balanced (within 1%)" },
  { value: "max", label: "Max quality" },
];
const STEP_MS = 700;

function ConfidenceBar({ p, th }: { p: number; th: number }) {
  const clamp = (x: number) => Math.max(0, Math.min(1, x));
  return (
    <div className="relative mt-2 h-2 w-full rounded-full bg-line" aria-hidden>
      <div className="h-2 rounded-full" style={{ width: `${clamp(p) * 100}%`, background: "var(--series-1)" }} />
      {th <= 1 && <div className="absolute -top-1 h-4 w-0.5 bg-ink" style={{ left: `${clamp(th) * 100}%` }} />}
    </div>
  );
}

function Receipt({ t }: { t: Trace }) {
  const saved = t.saved_joules_estimate;
  return (
    <Card title="Receipt">
      <div className="text-sm text-ink2">Answered by <span className="font-medium text-ink">{t.steps.at(-1)!.model}</span></div>
      <pre className="mt-2 max-h-56 overflow-auto whitespace-pre-wrap rounded-lg bg-page p-3 text-sm text-ink">{t.answer}</pre>
      {t.correct !== null && t.correct !== undefined && (
        <div className={`mt-2 text-sm font-medium ${t.correct ? "text-good" : "text-bad"}`}>
          {t.correct ? "✓ Matches the dataset answer" : "✗ Does not match the dataset answer"}
        </div>
      )}
      <dl className="num mt-3 grid grid-cols-2 gap-x-6 gap-y-1 text-sm sm:grid-cols-3">
        <div><dt className="text-ink2">Energy (all attempts)</dt><dd className="text-ink">{num(t.joules_total)} J</dd></div>
        <div><dt className="text-ink2">of which wasted</dt><dd className="text-ink">{num(t.wasted_joules)} J</dd></div>
        <div><dt className="text-ink2">Time</dt><dd className="text-ink">{num(t.seconds_total, 2)} s</dd></div>
        <div><dt className="text-ink2">Cost</dt><dd className="text-ink">₹{t.inr.toFixed(5)}</dd></div>
        <div><dt className="text-ink2">CO₂</dt><dd className="text-ink">{num(t.g_co2, 3)} g</dd></div>
        <div><dt className="text-ink2">vs largest model (est.)</dt>
          <dd className="text-ink">{saved >= 0 ? `${num(saved)} J saved` : `${num(-saved)} J more`}</dd></div>
      </dl>
      <p className="mt-3 text-xs text-muted">{t.notes}. Energy is {t.energy_kind} CPU package energy, net of idle.</p>
    </Card>
  );
}

export default function LiveLadder({ measuring }: { measuring: boolean }) {
  const [tasks, setTasks] = useState<TaskInfo | null>();
  const [err, setErr] = useState<string>();
  const [task, setTask] = useState("math");
  const [quality, setQuality] = useState("balanced");
  const [input, setInput] = useState<Record<string, string>>({});
  const [gold, setGold] = useState<unknown>(undefined);
  const [trace, setTrace] = useState<Trace>();
  const [shown, setShown] = useState(0);
  const [busy, setBusy] = useState(false);
  const [override, setOverride] = useState(false);
  const [openRef, setOpenRef] = useState<OpenRef | null>(null);
  useEffect(() => {
    api.openModels(task).then(setOpenRef).catch(() => setOpenRef(null));
  }, [task]);

  useEffect(() => {
    api.tasks().then((t) => { setTasks(t); if (t && !t[task]) setTask(Object.keys(t)[0]); }).catch((e) => setErr(String(e)));
  }, []);
  useEffect(() => {
    if (!trace) return;
    setShown(0);
    const timers = trace.steps.map((_, i) => setTimeout(() => setShown(i + 1), (i + 1) * STEP_MS));
    return () => timers.forEach(clearTimeout);
  }, [trace]);

  if (err) return <Empty title="Router not available" hint={err} />;
  if (tasks === undefined) return <div className="text-sm text-ink2">Loading the ladder (training confidence models from the cache)…</div>;
  if (!tasks || !Object.keys(tasks).length) return <Empty title="No calibrated ladder yet" hint="Run the eval cache first; the live router trains from it." />;

  const info = tasks[task];
  const blocked = measuring && !override;
  const run = async () => {
    setBusy(true); setTrace(undefined); setErr(undefined);
    try { setTrace(await api.ladder({ task, input, quality, gold })); }
    catch (e) { setErr(String(e)); }
    finally { setBusy(false); }
  };
  const example = async () => {
    const ex = await api.example(task);
    if (ex) { setInput(ex.input); setGold(ex.gold); setTrace(undefined); }
  };

  return (
    <div className="grid gap-4 lg:grid-cols-[minmax(0,5fr)_minmax(0,6fr)]">
      <Card title="Ask">
        <div className="space-y-3">
          <Chips label="Task" value={task} onChange={(v) => { setTask(v); setInput({}); setGold(undefined); setTrace(undefined); }}
                 options={Object.keys(tasks).map((t) => ({ value: t, label: t }))} />
          <Chips label="Quality" value={quality} onChange={setQuality} options={QUALITIES} />
          {info.input_fields.map((f) => (
            <label key={f} className="block text-sm text-ink2">
              {f}
              <textarea value={input[f] ?? ""} rows={f === "context" ? 6 : 3}
                        onChange={(e) => { setInput({ ...input, [f]: e.target.value }); setGold(undefined); }}
                        className="mt-1 w-full rounded-lg border border-line bg-page p-2 text-sm text-ink" />
            </label>
          ))}
          <div className="flex flex-wrap items-center gap-2">
            <button onClick={run} disabled={busy || blocked || !info.input_fields.every((f) => input[f]?.trim())}
                    className="rounded-lg bg-ink px-4 py-2 text-sm font-medium text-page disabled:opacity-40">
              {busy ? "Climbing…" : "Run the ladder"}
            </button>
            <button onClick={example} className="rounded-lg border border-line px-3 py-2 text-sm text-ink2 hover:bg-line/40">
              Load a test example
            </button>
          </div>
          {measuring && (
            <label className="flex items-center gap-2 text-xs text-ink2">
              <input type="checkbox" checked={override} onChange={(e) => setOverride(e.target.checked)} />
              Run anyway (adds load to the measurement in progress)
            </label>
          )}
          <p className="text-xs text-muted">Ladder: {info.tiers.map((t) => t.model).join(" → ")}</p>
        </div>
      </Card>
      <div className="space-y-4">
        {!trace && !busy && <Empty title="Nothing run yet" hint="Ask a question to watch it climb the ladder." />}
        {busy && <div className="text-sm text-ink2">Running real models on this CPU…</div>}
        {trace?.steps.slice(0, shown).map((s) => (
          <Card key={s.tier} title={`${s.tier} · ${s.model}`}
                right={<span className={`text-sm font-medium ${s.accepted ? "text-good" : "text-ink2"}`}>
                  {s.accepted ? (s.confidence === null ? "✓ Answered (top tier)" : "✓ Accepted") : "↑ Rejected, climbing"}
                </span>}>
            {s.confidence !== null && s.threshold !== null && (
              <>
                <div className="num text-sm text-ink2">confidence <span className="text-ink">{num(100 * s.confidence, 0)}%</span>{" "}
                  · needs {s.threshold > 1 ? "– (always climbs at this quality)" : `${num(100 * s.threshold, 0)}%`}</div>
                <ConfidenceBar p={s.confidence} th={s.threshold} />
              </>
            )}
            <div className="num mt-2 text-sm text-ink2">{num(s.joules)} J · {num(s.seconds, 2)} s · {s.tokens} tokens</div>
            {!s.accepted && <details className="mt-2 text-sm"><summary className="cursor-pointer text-ink2">Rejected answer</summary>
              <pre className="mt-1 whitespace-pre-wrap text-ink">{s.text}</pre></details>}
          </Card>
        ))}
        {trace && shown >= trace.steps.length && <Receipt t={trace} />}
        {openRef && <OpenModels r={openRef} />}
      </div>
    </div>
  );
}
