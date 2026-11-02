import { useState } from "react";
import { MODES, modePoint, type ModeKey, type Results } from "../api";
import { Card, Chips, Stat, Table, escalation, num, pct, signedPct } from "../ui";
import { ResultsGate, useResults } from "./Results";

function Knob({ data }: { data: Results }) {
  const targets = [...data.targets].sort((a, b) => a - b);
  const [i, setI] = useState(Math.max(0, targets.indexOf(0.01)));
  const [mode, setMode] = useState<ModeKey>("hybrid");
  const target = targets[i];
  const idx = (t: { ladder: { target: number }[] }) => t.ladder.findIndex((p) => p.target === target);

  const m = data.mixed;
  const mp = m.ladder[idx(m)];
  const mq = mp ? modePoint(mp, mode) : undefined;
  const big = m.baselines.always_largest;

  return (
    <div className="space-y-4">
      <Card title="Quality target">
        <label htmlFor="knob" className="text-sm text-ink2">
          Stay within <span className="font-semibold text-ink">{pct(target, 0)}</span> of always-largest accuracy (chosen on dev, shown on test)
        </label>
        <input id="knob" type="range" min={0} max={targets.length - 1} step={1} value={i}
               onChange={(e) => setI(Number(e.target.value))} className="mt-3 w-full accent-[var(--series-1)]" />
        <div className="mt-1 flex justify-between text-xs text-muted">
          <span>max quality</span><span>fast</span>
        </div>
        <div className="mt-4">
          <Chips label="Routing mode" value={mode} onChange={setMode}
                 options={MODES.map((x) => ({ value: x.key, label: x.label }))} />
        </div>
      </Card>
      {mq && big && (
        <div className="grid gap-3 sm:grid-cols-3">
          <Stat label="Mixed-traffic accuracy" value={pct(mq.test.accuracy)} sub={`always-largest ${pct(big.accuracy)}`} />
          <Stat label="Energy vs always-largest" value={signedPct(100 * (mq.test.joules_total / big.joules_total - 1))}
                sub={`${num(mq.test.joules_total / 1000)} kJ vs ${num(big.joules_total / 1000)} kJ`} />
          <Stat label="Cost for these requests" value={`₹${(mq.test.inr ?? 0).toFixed(4)}`}
                sub={`${num(mq.test.g_co2, 2)} g CO₂ at ₹${data.tariff.inr_per_kwh}/kWh, ${data.tariff.grid_kg_co2_per_kwh} kg/kWh`} />
        </div>
      )}
      <Card title="Per task">
        <Table
          head={["task", "accuracy", "Δ vs largest", "energy vs largest", "escalation per step", "wrong answers accepted early"]}
          rows={data.tasks.map((t) => {
            const p = t.ladder[idx(t)];
            const q = p ? modePoint(p, mode) ?? p : undefined;
            const b = t.baselines.always_largest;
            if (!q) return [t.task, "–", "–", "–", "–", "–"];
            return [t.task, pct(q.test.accuracy), signedPct(100 * (q.test.accuracy - b.accuracy)),
                    signedPct(100 * (q.test.joules_total / b.joules_total - 1)),
                    escalation(q, t.models),
                    pct(q.test.accepted_wrong_rate)];
          })}
        />
        <p className="mt-2 text-xs text-ink2">From stored test-split results; nothing is re-run when you move the slider.</p>
      </Card>
    </div>
  );
}

export default function QualityKnob() {
  const { data, error } = useResults();
  return <ResultsGate data={data} error={error}>{(d) => <Knob data={d} />}</ResultsGate>;
}
