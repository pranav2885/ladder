import type { OpenModel, OpenRef } from "./api";
import { Card, Table } from "./ui";

function fit(m: OpenModel) {
  if (m.fits_this_machine === null) return <span className="text-muted">size not stated</span>;
  return m.fits_this_machine
    ? <span className="text-good">✓ ~{m.est_q4_ram_gb} GB</span>
    : <span className="text-ink2">✗ ~{m.est_q4_ram_gb} GB</span>;
}

export default function OpenModels({ r }: { r: OpenRef }) {
  return (
    <Card
      title={`Best open-weight models for this kind of prompt · ${r.category_label}`}
      right={<span className="text-xs text-muted">reference only · never run</span>}
    >
      <div className="grid gap-3 sm:grid-cols-2">
        <div className="rounded-lg bg-page p-3">
          <div className="text-xs text-ink2">Top open model ({r.category_label})</div>
          <div className="mt-0.5 font-semibold text-ink">{r.best.model}</div>
          <div className="num text-xs text-ink2">{r.best.org} · {r.best.license} · #{r.best.rank} on the leaderboard · score {r.best.score}</div>
        </div>
        <div className="rounded-lg bg-page p-3">
          <div className="text-xs text-ink2">Best one that would fit this machine{r.ram_gb ? ` (${r.ram_gb} GB RAM)` : ""}</div>
          {r.best_that_fits ? (
            <>
              <div className="mt-0.5 font-semibold text-ink">{r.best_that_fits.model}</div>
              <div className="num text-xs text-ink2">{r.best_that_fits.org} · #{r.best_that_fits.rank} · ~{r.best_that_fits.est_q4_ram_gb} GB at 4-bit (estimate)</div>
            </>
          ) : <div className="mt-0.5 text-sm text-ink2">None of the listed models states a size that fits.</div>}
        </div>
      </div>
      <div className="mt-3">
        <Table
          head={["major lab", "best open model", "rank", "score", "licence", "fits here"]}
          rows={r.major_labs.map((m) => [m.org, m.model, `#${m.rank}`, m.score, m.license, fit(m)])}
        />
      </div>
      <p className="mt-2 text-xs text-muted">
        Source: <a className="underline" href={r.source_url} target="_blank" rel="noreferrer">{r.source}, {r.category_label}</a>,
        as of {r.as_of}{r.mapped_from_task ? ` (Ladder's “${r.mapped_from_task}” task is mapped to this category)` : ""}. {r.note}
      </p>
    </Card>
  );
}
