# Ladder results (quality target: within 1% of always-largest on dev)

Generated 2026-11-01T14:09:18.378798+00:00. Test split only; thresholds chosen on dev.

| task | ladder | always-largest acc | Ladder acc | Δ acc | energy saved vs largest | % of oracle savings | random-escalation acc | escalation rate per step |
|---|---|---|---|---|---|---|---|---|
| classify | qwen2.5:0.5b → qwen2.5:3b → qwen2.5:7b | 88.0% | 86.0% | -2.0 pt | 30.1% | 51.7% | 76.3% | 58%, 0% |
| extract | qwen2.5:0.5b → qwen2.5:3b → qwen2.5:7b | 99.0% | 98.0% | -1.0 pt | -29.9% | -58.6% | 93.8% | 98%, 41% |
| qa | qwen2.5:0.5b → qwen2.5:3b → qwen2.5:7b | 82.0% | 65.0% | -17.0 pt | 27.5% | 52.6% | 54.5% | 38%, 26% |
| math | qwen2.5:0.5b → qwen2.5:3b → qwen2.5:7b | 92.0% | 91.0% | -1.0 pt | -36.2% | -68.8% | 87.6% | 97%, 54% |

### Routing modes compared (test accuracy · energy saved vs always-largest)

| task | cascade | pre-routing | hybrid |
|---|---|---|---|
| classify | 86.0% · 30.1% | 86.0% · 36.4% | 83.0% · 39.6% |
| extract | 98.0% · -29.9% | 96.0% · 20.9% | 98.0% · 15.4% |
| qa | 65.0% · 27.5% | 70.0% · 26.4% | 70.0% · 23.9% |
| math | 91.0% · -36.2% | 89.0% · 22.9% | 87.0% · 14.8% |

**Mixed traffic (400 requests):** Ladder 85.0% vs always-largest 90.2%, using -29.0% less CPU energy (133.5 kJ vs 103.5 kJ; ₹0.2968 vs ₹0.2301).

- Pre-routing: 85.2% accuracy, 23.0% less energy than always-largest
- Hybrid: 84.5% accuracy, 16.1% less energy than always-largest

Note: 'largest' means the largest tier with complete cache coverage for that task; see the ladder column.

Energy: CPU package power from LibreHardwareMonitor, minus the idle baseline, including the work of every rejected attempt. Not wall power.
