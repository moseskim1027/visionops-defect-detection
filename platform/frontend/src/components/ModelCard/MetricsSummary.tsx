import type { ModelCard, OverallMetrics } from '../../types'

interface Props {
  card: ModelCard | null
  loading: boolean
}

export default function MetricsSummary({ card, loading }: Props) {
  if (loading) {
    return (
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        {[...Array(4)].map((_, i) => (
          <div key={i} className="h-24 bg-slate-800 rounded-xl animate-pulse" />
        ))}
      </div>
    )
  }

  const m: OverallMetrics = card?.metrics ?? { precision: 0, recall: 0, map50: 0, map50_95: 0 }
  const tiles = [
    { label: 'Precision', value: m.precision, color: 'green' },
    { label: 'Recall', value: m.recall, color: 'cyan' },
    { label: 'mAP50', value: m.map50, color: 'indigo' },
    { label: 'mAP50-95', value: m.map50_95, color: 'purple' },
  ]

  const startTime = card?.start_time ? new Date(card.start_time).toLocaleString() : null
  const endTime = card?.end_time ? new Date(card.end_time).toLocaleString() : null

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        {tiles.map(t => (
          <MetricTile key={t.label} {...t} />
        ))}
      </div>

      {card && (
        <div className="bg-slate-900 rounded-xl border border-slate-800 p-4 flex flex-wrap gap-4 text-xs text-slate-400">
          <span><span className="text-slate-500">Run: </span>
            <code className="text-slate-300">{card.run_id?.slice(0, 16)}…</code>
          </span>
          {startTime && <span><span className="text-slate-500">Started: </span>{startTime}</span>}
          {endTime && <span><span className="text-slate-500">Completed: </span>{endTime}</span>}
          {card.params?.epochs && (
            <span><span className="text-slate-500">Epochs: </span>{card.params.epochs}</span>
          )}
          {card.params?.batch && (
            <span><span className="text-slate-500">Batch: </span>{card.params.batch}</span>
          )}
          {card.params?.lr0 && (
            <span><span className="text-slate-500">LR₀: </span>{card.params.lr0}</span>
          )}
        </div>
      )}
    </div>
  )
}

function MetricTile({
  label, value, color,
}: {
  label: string; value: number; color: string
}) {
  const pct = Math.round(value * 100)
  const ringColor: Record<string, string> = {
    green: 'from-green-500/20 to-green-500/5 border-green-800/40',
    cyan: 'from-cyan-500/20 to-cyan-500/5 border-cyan-800/40',
    indigo: 'from-indigo-500/20 to-indigo-500/5 border-indigo-800/40',
    purple: 'from-purple-500/20 to-purple-500/5 border-purple-800/40',
  }
  const textColor: Record<string, string> = {
    green: 'text-green-400',
    cyan: 'text-cyan-400',
    indigo: 'text-indigo-400',
    purple: 'text-purple-400',
  }
  return (
    <div className={`bg-gradient-to-br ${ringColor[color]} border rounded-xl p-4 space-y-1`}>
      <div className="text-xs text-slate-400">{label}</div>
      <div className={`text-3xl font-bold ${textColor[color]}`}>{pct}%</div>
      <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full bg-current ${textColor[color]}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}
