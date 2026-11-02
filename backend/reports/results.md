# Ladder results (quality target: within 1% of always-largest on dev)

Generated 2026-11-02T03:28:54.060077+00:00. Test split only; thresholds chosen on dev.

| task | ladder | always-largest acc | Ladder acc | Δ acc | energy saved vs largest | % of oracle savings | random-escalation acc | escalation rate per step |
|---|---|---|---|---|---|---|---|---|
| classify | qwen2.5:0.5b → qwen2.5:3b → qwen2.5:7b | 88.0% | 88.0% | +0.0 pt | 29.8% | 51.1% | 88.0% | starts at qwen2.5:3b; 0% |
| extract | qwen2.5:0.5b → qwen2.5:3b → qwen2.5:7b | 99.0% | 99.0% | +0.0 pt | 0.0% | 0.0% | 99.0% | starts at qwen2.5:7b; – |
| qa | qwen2.5:0.5b → qwen2.5:3b → qwen2.5:7b | 82.0% | 65.0% | -17.0 pt | 27.5% | 52.6% | 54.5% | 38%, 26% |
| math | qwen2.5:0.5b → qwen2.5:3b → qwen2.5:7b | 92.0% | 92.0% | +0.0 pt | 0.0% | 0.0% | 92.0% | starts at qwen2.5:7b; – |

### Routing modes compared (test accuracy · energy saved vs always-largest)

| task | cascade | pre-routing | hybrid |
|---|---|---|---|
| classify | 88.0% · 29.8% | 86.0% · 36.4% | 83.0% · 39.6% |
| extract | 99.0% · 0.0% | 96.0% · 20.9% | 98.0% · 15.4% |
| qa | 65.0% · 27.5% | 70.0% · 26.4% | 70.0% · 23.9% |
| math | 92.0% · 0.0% | 89.0% · 22.9% | 87.0% · 14.8% |

**Mixed traffic (400 requests):** Ladder 86.0% vs always-largest 90.2%, using 2.6% less CPU energy (100.9 kJ vs 103.5 kJ; ₹0.2242 vs ₹0.2301).

- Pre-routing: 85.2% accuracy, 23.0% less energy than always-largest
- Hybrid: 84.5% accuracy, 16.1% less energy than always-largest

Note: 'largest' means the largest tier with complete cache coverage for that task; see the ladder column.

Energy: CPU package power from LibreHardwareMonitor, minus the idle baseline, including the work of every rejected attempt. Not wall power.
