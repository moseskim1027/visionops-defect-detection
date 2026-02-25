import type { PoorSample } from '../../types'

interface Props {
  samples: PoorSample[]
  loading: boolean
}

export default function PoorSamples({ samples, loading }: Props) {
  if (loading) {
    return (
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
        {Array.from({ length: 8 }).map((_, i) => (
          <div key={i} className="aspect-square bg-slate-800 rounded-lg animate-pulse" />
        ))}
      </div>
    )
  }

  if (!samples.length) {
    return (
      <div className="bg-slate-900 rounded-xl border border-slate-800 p-8 text-center text-slate-500 text-sm">
        No poor-performing samples — run validation or train the model first.
      </div>
    )
  }

  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
      {samples.map((sample, i) => (
        <div
          key={i}
          className="group relative bg-slate-800 rounded-xl overflow-hidden border border-slate-700 hover:border-red-600/60 transition-colors"
        >
          <img
            src={`data:image/jpeg;base64,${sample.image_b64}`}
            alt={sample.filename}
            className="w-full aspect-square object-cover"
            loading="lazy"
          />
          {/* Gradient overlay */}
          <div className="absolute inset-0 bg-gradient-to-t from-black/90 via-black/20 to-transparent" />

          {/* Bottom info */}
          <div className="absolute bottom-0 left-0 right-0 p-2.5">
            <div className="text-white text-xs font-medium truncate">{sample.class_name}</div>
            <div className="flex items-center gap-1.5 mt-0.5">
              <APBadge ap={sample.ap50} />
              <span className="text-slate-400 text-xs truncate">{sample.filename}</span>
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}

function APBadge({ ap }: { ap: number }) {
  const pct = (ap * 100).toFixed(0)
  const cls =
    ap >= 0.7 ? 'bg-green-600/80 text-green-100' :
    ap >= 0.4 ? 'bg-amber-600/80 text-amber-100' :
    'bg-red-600/80 text-red-100'
  return (
    <span className={`text-xs font-bold px-1.5 py-0.5 rounded flex-shrink-0 ${cls}`}>
      {pct}%
    </span>
  )
}
