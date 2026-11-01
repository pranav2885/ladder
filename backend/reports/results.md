# Ladder results (quality target: within 1% of always-largest on dev)

Generated 2026-11-01T03:38:52.661094+00:00. Test split only; thresholds chosen on dev.

| task | ladder | always-largest acc | Ladder acc | Δ acc | energy saved vs largest | % of oracle savings | random-escalation acc | escalation rate per step |
|---|---|---|---|---|---|---|---|---|
| classify | qwen2.5:0.5b → qwen2.5:3b | 88.0% | 86.0% | -2.0 pt | 0.5% | 1.2% | 76.3% | 58% |
| extract | qwen2.5:0.5b → qwen2.5:3b | 92.0% | 91.0% | -1.0 pt | -24.2% | -66.6% | 91.0% | 98% |
| qa | qwen2.5:0.5b → qwen2.5:3b | 70.0% | 69.0% | -1.0 pt | -16.4% | -33.8% | 65.2% | 82% |
| math | qwen2.5:0.5b | 36.0% | 36.0% | +0.0 pt | 0.0% | nan% | – | – |

### Routing modes compared (test accuracy · energy saved vs always-largest)

| task | cascade | pre-routing | hybrid |
|---|---|---|---|
| classify | 86.0% · 0.5% | 86.0% · 9.5% | 83.0% · 14.0% |
| extract | 91.0% · -24.2% | 90.0% · 2.3% | 91.0% · 13.9% |
| qa | 69.0% · -16.4% | 70.0% · 0.0% | 70.0% · 0.0% |
| math | 36.0% · 0.0% | – | – |

**Mixed traffic (400 requests):** Ladder 70.5% vs always-largest 71.5%, using -12.0% less CPU energy (44.1 kJ vs 39.4 kJ; ₹0.0980 vs ₹0.0875).

- Pre-routing: 70.5% accuracy, 1.3% less energy than always-largest
- Hybrid: 70.0% accuracy, 6.1% less energy than always-largest

Note: 'largest' means the largest tier with complete cache coverage for that task; see the ladder column.

Energy: CPU package power from LibreHardwareMonitor, minus the idle baseline, including the work of every rejected attempt. Not wall power.
