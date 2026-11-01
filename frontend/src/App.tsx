import { type ComponentType, useEffect, useState } from "react";
import { api } from "./api";
import LiveLadder from "./pages/LiveLadder";
import Machine from "./pages/Machine";
import Methodology from "./pages/Methodology";
import QualityKnob from "./pages/QualityKnob";
import Results from "./pages/Results";

const PAGES: { id: string; label: string; el: ComponentType<{ measuring: boolean }> }[] = [
  { id: "live", label: "Live Ladder", el: LiveLadder },
  { id: "results", label: "Results", el: Results },
  { id: "knob", label: "Quality Knob", el: QualityKnob },
  { id: "machine", label: "Machine", el: Machine },
  { id: "method", label: "Methodology", el: Methodology },
];

function usePage() {
  const read = () => (window.location.hash.slice(1) || "results").split("?")[0];
  const [page, setPage] = useState(read);
  useEffect(() => {
    const on = () => setPage(read());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return page;
}

export default function App() {
  const page = usePage();
  const [measuring, setMeasuring] = useState(false);
  useEffect(() => {
    const poll = () => api.health().then((h) => setMeasuring(!!h?.measurement_in_progress)).catch(() => {});
    poll();
    const id = setInterval(poll, 15000);
    return () => clearInterval(id);
  }, []);
  const Current = (PAGES.find((p) => p.id === page) ?? PAGES[1]).el;

  return (
    <div className="mx-auto max-w-6xl px-4 pb-16">
      <header className="flex flex-wrap items-center justify-between gap-3 py-5">
        <div>
          <div className="text-lg font-semibold text-ink">Ladder</div>
          <div className="text-xs text-ink2">Smallest model first. Climb only when needed. CPU only.</div>
        </div>
        <nav className="flex flex-wrap gap-1" aria-label="Pages">
          {PAGES.map((p) => (
            <a
              key={p.id}
              href={`#${p.id}`}
              aria-current={p.id === page ? "page" : undefined}
              className={`rounded-lg px-3 py-1.5 text-sm ${
                p.id === page ? "bg-ink text-page" : "text-ink2 hover:bg-line/50"
              }`}
            >
              {p.label}
            </a>
          ))}
        </nav>
      </header>
      {measuring && (
        <div role="status" className="mb-4 rounded-lg border border-line bg-surface px-4 py-2 text-sm text-ink2">
          <span className="font-medium text-warn">⚠ Measurement run in progress.</span> The eval cache is
          recording energy right now; live requests would add CPU load to those readings, so Live Ladder is
          paused until it finishes.
        </div>
      )}
      <Current measuring={measuring} />
    </div>
  );
}
