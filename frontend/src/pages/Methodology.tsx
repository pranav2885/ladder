import { Card } from "../ui";

const sections: [string, string[]][] = [
  ["What Ladder does", [
    "Every request goes to the smallest local model first. Ladder checks how likely that answer is correct and only climbs to a bigger model when it has to. All inference runs locally on the CPU with open-weight Qwen2.5 models through Ollama.",
  ]],
  ["Three routing modes", [
    "Cascade: run the small model, score its answer, escalate if the score is below the tier's threshold.",
    "Pre-routing: before any model runs, a small classifier looks only at the request (length, numbers, multi-step wording) and predicts the smallest tier likely to get it right. Nothing else runs, so obviously hard requests skip the wasted small-model attempt.",
    "Hybrid: start at the pre-routed tier, then cascade upward on answer confidence.",
  ]],
  ["Confidence signals", [
    "Token log-probabilities of the answer (mean, minimum, entropy over the top-5 alternatives, the share of low-probability tokens, and the same for the final-answer tokens).",
    "Free validators: JSON parses and matches the schema, the label is in the allowed set, a final number is present, the answer appears in the context, the output was not cut off.",
    "These are combined into one P(correct) per task and tier with logistic regression trained on the dev split.",
  ]],
  ["Calibration and the quality knob", [
    "Each dataset is split into dev (100 items per task) and test (100 per task). Thresholds are chosen on dev only: the cheapest setting whose dev accuracy stays within the target (0–10%) of always using the largest model. Dev probabilities are out-of-fold so thresholds aren't tuned on the classifier's own training fit.",
    "Every number on the Results and Quality Knob pages is from the test split, which is never used for tuning.",
  ]],
  ["Counting the full cost", [
    "When a small model's answer is rejected, its energy and time still count. Ladder's totals include every attempt, and the wasted part is reported separately.",
    "Baselines on the same machine: always-smallest, always-medium, always-largest; random escalation at the same rate as the cascade (shows whether the confidence signal matters); and the oracle, which sends each item straight to the cheapest tier that gets it right (the ceiling no real router can beat).",
  ]],
  ["Energy measurement", [
    "CPU package power from LibreHardwareMonitor (the Ryzen's own energy counters), sampled four times a second and integrated per request, minus a measured idle baseline. This is CPU package energy, not wall power.",
    "Requests run one at a time at temperature 0 after warm-up runs. Runs stop automatically if the laptop goes on battery, if power drops below idle during generation (a sensor or power-state change), or if any model layer is on the GPU.",
    "Rupees use ₹8/kWh and CO₂ uses 0.7 kg/kWh by default; both are editable in ladder.yaml.",
  ]],
  ["Limitations", [
    "Confidence signals are imperfect: some wrong answers are accepted confidently. The rate is reported as 'wrong answers accepted early'.",
    "Savings depend on the traffic mix. A workload of only hard questions saves little or costs more, because the small model's attempt is wasted.",
    "Results come from one laptop (Ryzen 7 4800H) with some background load (NVIDIA's FrameView service), which is included in the idle baseline.",
    "The 'saved vs largest' figure on a live request is an estimate: the median energy the largest model used on that task in the eval cache, not a run of the largest model on that request.",
    "Keeping all tiers loaded costs memory (about 8 GB for the three Qwen2.5 models).",
  ]],
];

export default function Methodology() {
  return (
    <div className="space-y-4">
      {sections.map(([title, paras]) => (
        <Card key={title} title={title}>
          <div className="max-w-3xl space-y-2 text-sm leading-relaxed text-ink2">
            {paras.map((p, i) => <p key={i}>{p}</p>)}
          </div>
        </Card>
      ))}
    </div>
  );
}
