import type { ReactNode } from "react";

export const pct = (x: number | undefined | null, d = 1) =>
  x === undefined || x === null || Number.isNaN(x) ? "–" : `${(100 * x).toFixed(d)}%`;
export const num = (x: number | undefined | null, d = 1) =>
  x === undefined || x === null || Number.isNaN(x) ? "–" : x.toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: d });
export const signedPct = (x: number | undefined | null, d = 1) =>
  x === undefined || x === null || Number.isNaN(x) ? "–" : `${x > 0 ? "+" : ""}${x.toFixed(d)}%`;

export function Card({ title, children, right }: { title?: string; children: ReactNode; right?: ReactNode }) {
  return (
    <section className="rounded-xl border border-line bg-surface p-4 sm:p-5">
      {(title || right) && (
        <div className="mb-3 flex flex-wrap items-baseline justify-between gap-2">
          {title && <h2 className="text-sm font-semibold text-ink">{title}</h2>}
          {right}
        </div>
      )}
      {children}
    </section>
  );
}

export function Stat({ label, value, sub }: { label: string; value: ReactNode; sub?: ReactNode }) {
  return (
    <div className="rounded-xl border border-line bg-surface p-4">
      <div className="text-xs text-ink2">{label}</div>
      <div className="mt-1 text-2xl font-semibold text-ink">{value}</div>
      {sub && <div className="mt-1 text-xs text-muted">{sub}</div>}
    </div>
  );
}

export function Empty({ title, hint }: { title: string; hint: ReactNode }) {
  return (
    <div className="rounded-xl border border-dashed border-line bg-surface p-8 text-center">
      <div className="font-medium text-ink">{title}</div>
      <div className="mt-1 text-sm text-ink2">{hint}</div>
    </div>
  );
}

export function Chips<T extends string>({ value, options, onChange, label }: {
  value: T;
  options: { value: T; label: string }[];
  onChange: (v: T) => void;
  label: string;
}) {
  return (
    <div role="radiogroup" aria-label={label} className="flex flex-wrap gap-1">
      {options.map((o) => (
        <button
          key={o.value}
          role="radio"
          aria-checked={o.value === value}
          onClick={() => onChange(o.value)}
          className={`rounded-full border px-3 py-1 text-sm ${
            o.value === value ? "border-ink bg-ink text-page" : "border-line text-ink2 hover:bg-line/40"
          }`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Table({ head, rows }: { head: ReactNode[]; rows: ReactNode[][] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-line text-left text-xs text-ink2">
            {head.map((h, i) => (
              <th key={i} className={`py-2 pr-4 font-medium ${i ? "text-right" : ""}`}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="border-b border-line/60 last:border-0">
              {r.map((c, j) => (
                <td key={j} className={`num py-2 pr-4 ${j ? "text-right" : "text-ink"} text-ink`}>{c}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function Swatch({ color, shape = "dot" }: { color: string; shape?: "dot" | "line" }) {
  return shape === "line" ? (
    <span className="inline-block h-0.5 w-4 rounded" style={{ background: color }} />
  ) : (
    <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: color }} />
  );
}
