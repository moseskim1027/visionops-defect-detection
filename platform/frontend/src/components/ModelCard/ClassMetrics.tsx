import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, ReferenceLine,
} from 'recharts'
import type { ClassMetric } from '../../types'

interface Props {
  classMetrics: ClassMetric[]
  loading: boolean
  selectedClass?: string | null
  onClassClick?: (name: string | null) => void
}

const AP_COLOR = (ap: number) => {
  if (ap >= 0.7) return '#22c55e'
  if (ap >= 0.4) return '#f59e0b'
  return '#ef4444'
}

export default function ClassMetrics({ classMetrics, loading, selectedClass, onClassClick }: Props) {
  const displayed = classMetrics.slice(0, 30)

  return (
    <div className="bg-slate-900 rounded-xl border border-slate-800 p-5 flex flex-col h-full">
      {/* Header */}
      <div className="flex-shrink-0 mb-3">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-slate-200 uppercase tracking-wider">
            Per-class AP50
          </h3>
          {selectedClass && onClassClick && (
            <button
              onClick={() => onClassClick(null)}
              className="text-xs text-slate-500 hover:text-slate-300 transition-colors"
            >
              Clear filter
            </button>
          )}
        </div>
        <p className="text-xs text-slate-500 mt-0.5">
          {onClassClick ? 'Click a class to filter samples' : 'Sorted ascending — worst classes first'}
        </p>
      </div>

      {loading ? (
        <div className="flex-1 bg-slate-800 rounded-lg animate-pulse" />
      ) : classMetrics.length === 0 ? (
        <div className="flex-1 flex items-center justify-center text-slate-600 text-sm">
          No metrics — train a model first
        </div>
      ) : (
        <div className="flex-1 min-h-0 flex flex-col">
          <div className="flex-1 min-h-0">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={displayed}
                layout="vertical"
                margin={{ top: 0, right: 24, left: 4, bottom: 0 }}
              >
                <XAxis
                  type="number"
                  domain={[0, 1]}
                  tick={{ fill: '#64748b', fontSize: 10 }}
                  axisLine={false}
                  tickLine={false}
                />
                <YAxis
                  type="category"
                  dataKey="class_name"
                  width={90}
                  tick={{ fill: '#94a3b8', fontSize: 10 }}
                  axisLine={false}
                  tickLine={false}
                />
                <Tooltip
                  cursor={false}
                  contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8, fontSize: 11 }}
                  labelStyle={{ color: '#f1f5f9' }}
                  itemStyle={{ color: '#94a3b8' }}
                  formatter={(v: number) => [`${(v * 100).toFixed(1)}%`, 'AP50']}
                />
                <ReferenceLine x={0.5} stroke="#334155" strokeDasharray="3 3" />
                <Bar
                  dataKey="ap50"
                  radius={[0, 4, 4, 0]}
                  maxBarSize={14}
                  onClick={onClassClick ? (data) => {
                    onClassClick(data.class_name === selectedClass ? null : data.class_name)
                  } : undefined}
                  style={{ cursor: onClassClick ? 'pointer' : 'default' }}
                >
                  {displayed.map((d, i) => (
                    <Cell
                      key={i}
                      fill={AP_COLOR(d.ap50)}
                      fillOpacity={!selectedClass || d.class_name === selectedClass ? 0.85 : 0.25}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Legend */}
          <div className="flex items-center gap-4 text-xs text-slate-400 pt-2 flex-shrink-0">
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-sm bg-green-500 inline-block" /> ≥70%
            </span>
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-sm bg-amber-500 inline-block" /> 40–70%
            </span>
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-sm bg-red-500 inline-block" /> &lt;40%
            </span>
          </div>
        </div>
      )}
    </div>
  )
}
