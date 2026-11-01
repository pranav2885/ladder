# Ladder

**Don't use a 7B model to answer a 1B question.** Ladder sends every request to the smallest local
model first, checks how confident the answer is, and only climbs to a bigger model when it has to.
All inference is local, on CPU, with open-weight models.

Built for Ziroh Labs Build Next 2026 (Kompact AI).

## Results

Ladder qwen2.5 0.5B → 3B → 7B on a Ryzen 7 4800H (CPU only), 100 test items per task. Thresholds
were chosen on the dev split only, against a target of staying within 1% of always-largest accuracy.
Energy is CPU package energy net of idle, and includes every rejected attempt. Full tables:
[`backend/reports/results.md`](backend/reports/results.md).

| task | always-7B acc | cascade | pre-routing | hybrid |
|---|---|---|---|---|
| classify | 88.0% | 86.0% · 30.1% saved | 86.0% · 36.4% saved | 83.0% · 39.6% saved |
| extract | 99.0% | 98.0% · −29.9% | 96.0% · 20.9% saved | 98.0% · 15.4% saved |
| qa | 82.0% | 65.0% · 27.5% saved | 70.0% · 26.4% saved | 70.0% · 23.9% saved |
| math | 92.0% | 91.0% · −36.2% | 89.0% · 22.9% saved | 87.0% · 14.8% saved |
| **mixed (400)** | **90.2%** | 85.0% · −29.0% | **85.2% · 23.0% saved** | 84.5% · 16.1% saved |

What this shows:

- **Pre-routing is the mode that pays off.** It picks a tier from the prompt alone and saves 23% energy
  on mixed traffic for 5 points of accuracy.
- **Cascading costs more than it saves on math and extract.** 0.5B is escalated on 97–98% of those
  items, so the wasted first attempts make the cascade 30–36% *more* expensive than running 7B.
- **QA misses the 1% target on test.** On dev, 7B (78%) was no better than 3B (79%), so the
  thresholds chosen there did not carry over.
- Confidence-based escalation beats random escalation at the same rate on every task.

**Caveat: 7B energy is estimated.** From 1 Nov 2026 the laptop's firmware capped the CPU at about
12 W, which affected most 7B math and QA rows and some 7B extract rows. Those runs kept their
answers (temperature 0, fixed seed), but the eval replaces their energy and time with a per-tier
least-squares fit over the same tier's full-power rows (joules ~ prompt + completion tokens),
labelled `estimated-from-full-power`. The energy repetitions were also skipped (`--reps 1`), so
there are no per-item energy medians.

## Status

| Piece | State |
|---|---|
| Backend adapters (Ollama, llama.cpp) with token logprobs | done |
| Energy: LibreHardwareMonitor (Windows), RAPL (Linux), estimator fallback, idle baseline | done |
| Datasets: classify / extract / qa / math, fixed dev/test splits | done |
| Item x tier cache (resumable, power-state guards, capped mode) | done |
| Confidence signals, cascade, calibration, baselines | done |
| Pre-routing, hybrid, router API (OpenAI-compatible), dashboard | done |

## Setup (Windows)

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\pip install -e ".[dev]"
.\.venv\Scripts\ladder doctor
```

1. **Models** (Ollama): `ollama pull qwen2.5:0.5b`, `ollama pull qwen2.5:3b`, `ollama pull qwen2.5:7b`
   (about 7 GB on disk; about 8–9 GB RAM to keep all three loaded).
2. **Energy**: install [LibreHardwareMonitor](https://github.com/LibreHardwareMonitor/LibreHardwareMonitor),
   run it **as administrator**, then *Options → Remote Web Server → Run* (port 8085).
   `ladder energy probe` should then print CPU package watts. Without it Ladder falls back to a
   TDP x utilisation estimate and labels every number `estimated`.
   On Linux, RAPL is used instead (`sudo chmod o+r /sys/class/powercap/intel-rapl:*/energy_uj`).
3. **Idle baseline**: close other apps, plug in the charger, then `ladder energy idle --seconds 60`.
4. **Data**: `ladder data fetch` then `ladder data build`.
5. **Cache**: `ladder cache run` (several hours on a laptop CPU; safe to stop and resume), then
   `ladder cache status`. Try `ladder cache run --limit 3 --reps 1` first.

## Evaluate, report, serve

```powershell
.\.venv\Scripts\ladder eval      # calibrate on dev for each quality target, evaluate on test, write reports/
.\.venv\Scripts\ladder serve     # router API + dashboard at http://127.0.0.1:8000
```

- `reports/results.md|csv|json` and `accuracy_vs_energy_*.png`: every baseline and routing mode.
- Dashboard pages: **Live Ladder** (ask a question and watch it climb), **Results**,
  **Quality Knob**, **Machine**, **Methodology**. Pages show empty states until real runs exist.
- OpenAI-compatible endpoint: `POST /v1/chat/completions` with `model: "ladder-<task>"`
  (`ladder-math`, `ladder-classify`, `ladder-extract`, `ladder-qa` with "Context: … Question: …").
  The response carries a `ladder` routing trace and `x-ladder-tier`, `x-ladder-joules` and
  `x-ladder-saved-joules` headers. With `stream: true` the answer arrives as one chunk, because it
  is only known after the confidence check.
- Frontend dev: `cd frontend && npm install && npm run dev` (proxies `/api` to :8000);
  `npm run build` produces `frontend/dist`, which `ladder serve` serves.
- Live requests are paused while `ladder cache run` is measuring, so they don't distort its readings.
- **Open-weight reference.** For each prompt, Live Ladder and the API (`open_model_reference` in
  the trace, or `GET /api/open-models?task=…`) also show the strongest *open-weight, downloadable*
  models for that kind of prompt: the top model, the best model per major lab (Google, Meta,
  Alibaba, Mistral, Microsoft, OpenAI, DeepSeek, Nvidia, …) and the best one that would fit this
  machine's RAM (rough 4-bit estimate, only when the model name states its size). The data is a
  dated snapshot of the LMArena text leaderboard filtered to non-proprietary licences
  (`backend/ladder/data/open_models_arena_2026-10-28.json`, with source URLs). These models are a
  reference only: Ladder never downloads, calls or runs them, and no hosted AI API is used.

## Datasets

| Task | Source | License | Metric |
|---|---|---|---|
| classify | Banking77 test, 10-intent balanced subset | CC BY 4.0 | accuracy |
| extract | synthetic invoices generated from a fixed seed | n/a | field exact match (correct = all 5 fields) |
| qa | SQuAD v1.1 dev, one question per paragraph | CC BY-SA 4.0 | EM / token F1 (correct = EM or F1 ≥ 0.8) |
| math | GSM8K test | MIT | final-answer exact match |

Splits: 100 dev (calibration only) + 100 test (reporting only) per task, seed in `ladder.yaml`.

## Measurement notes

- Energy is **CPU package** power (LHM/RAPL), not wall power. LHM refreshes about once per second,
  so very short requests carry more relative error.
- Every cached row stores gross joules, net-of-idle joules, the idle baseline used, the source and
  whether it was `measured` or `estimated`.
- **Background load on the reference machine.** The dev laptop (Ryzen 7 4800H, on AC) had NVIDIA's
  FrameView service running `presentmon_x64` (~2 threads) plus Windows Defender during the runs.
  The idle baseline (29.0 W measured by LHM, CPU ~25%) includes that load, and `joules_net`
  subtracts it, but it adds noise and makes the baseline higher than a clean idle machine.
- **CPU-only is enforced.** Ollama offloads layers to any GPU it finds (on the dev laptop it put
  45% of the 7B model on an RTX 3050). Ladder sends `num_gpu: 0` with every request, includes
  backend options in the cache key, and aborts a run if `/api/ps` shows any model in GPU memory.
- Generations run one at a time at temperature 0; each tier is warmed up first and those runs are
  discarded. A subset of items can be repeated (default 3 runs) for energy medians; the published
  run used `--reps 1`.
- **Power-state guards.** A run stops when the laptop goes on battery, or when package power drops
  below the idle baseline (this caught a 45 W → 14 W drop on 28 Oct; those rows were quarantined).
  `cache run --power-capped` lets a run continue under a known cap, and those rows' energy is
  estimated as described under Results.
- All assumptions (tiers, tariff ₹/kWh, grid kg CO₂/kWh, TDP, sample rate) live in
  `backend/ladder.yaml`.
