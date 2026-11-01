# Ladder

**Don't use a 7B model to answer a 1B question.** Ladder sends every request to the smallest local
model first, checks how confident the answer is, and only climbs to a bigger model when it has to.
All inference is local, on CPU, with open-weight models.

Built for Ziroh Labs Build Next 2026 (Kompact AI). Results tables appear here only once they come
from real runs.

## Status

| Piece | State |
|---|---|
| Backend adapters (Ollama, llama.cpp) with token logprobs | done |
| Energy: LibreHardwareMonitor (Windows), RAPL (Linux), estimator fallback, idle baseline | done |
| Datasets: classify / extract / qa / math, fixed dev/test splits | done |
| Item x tier cache (resumable, energy repetitions) | done |
| Confidence signals, cascade, calibration, baselines | next |
| Pre-routing, router API, dashboard | later |

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
  discarded. A subset of items is repeated (default 3 runs) for energy medians.
- All assumptions (tiers, tariff ₹/kWh, grid kg CO₂/kWh, TDP, sample rate) live in
  `backend/ladder.yaml`.
