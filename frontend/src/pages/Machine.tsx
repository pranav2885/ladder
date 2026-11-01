import { useEffect, useState } from "react";
import { api } from "../api";
import { Card, Empty, Table, num } from "../ui";

export default function Machine() {
  const [m, setM] = useState<any>();
  const [err, setErr] = useState<string>();
  useEffect(() => {
    api.machine().then(setM).catch((e) => setErr(String(e)));
  }, []);
  if (err) return <Empty title="Could not load machine info" hint={err} />;
  if (!m) return <div className="text-sm text-ink2">Loading…</div>;
  const live = m.live;
  const tiers: string[] = m.tiers.map((t: any) => t.name);
  return (
    <div className="grid gap-4 md:grid-cols-2">
      <Card title="Hardware">
        <Table head={["", ""]} rows={[
          ["CPU", live.cpu],
          ["Cores / threads", `${live.physical_cores} / ${live.logical_cores}`],
          ["RAM", `${live.ram_gb} GB`],
          ["OS", live.os],
          ["On charger", live.power_plugged === null ? "n/a" : live.power_plugged ? "yes" : "no — measurements are blocked on battery"],
        ]} />
      </Card>
      <Card title="Energy measurement">
        {m.energy.error ? <div className="text-sm text-bad">{m.energy.error}</div> : (
          <Table head={["", ""]} rows={[
            ["Source", `${m.energy.source} (${m.energy.kind})`],
            ["Scope", m.energy.scope],
            ["Current reading", `${num(m.energy.reading)} ${m.energy.unit}`],
            ["Idle baseline", m.idle ? `${num(m.idle.idle_watts)} W, ${m.idle.energy_source}, CPU ${num(m.idle.mean_cpu_percent, 0)}% (${new Date(m.idle.created_at).toLocaleString()})` : "not calibrated"],
            ["Tariff", `₹${m.tariff.inr_per_kwh}/kWh · ${m.tariff.grid_kg_co2_per_kwh} kg CO₂/kWh`],
          ]} />
        )}
        {m.idle?.warning && <p className="mt-2 text-xs text-ink2">⚠ {m.idle.warning}</p>}
      </Card>
      <Card title="Model ladder">
        <Table head={["tier", "model", "backend"]} rows={m.tiers.map((t: any) => [t.name, t.model, t.backend])} />
        <p className="mt-2 text-xs text-ink2">CPU-only: every request sets num_gpu = 0, and runs abort if any layer sits in GPU memory.</p>
      </Card>
      <Card title="Eval cache coverage (main runs)">
        <Table head={["task / split", ...tiers]} rows={Object.entries(m.coverage).sort().map(([k, v]: any) => [
          k, ...tiers.map((t) => (v[t] ? `${v[t].n}` : "–")),
        ])} />
      </Card>
    </div>
  );
}
