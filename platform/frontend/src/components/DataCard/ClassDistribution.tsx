import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
} from 'recharts'
import type { ClassCount } from '../../types'

interface Props {
  distribution: ClassCount[]
  loading: boolean
  selectedClass?: string | null
  onClassClick?: (name: string | null) => void
}

const COLORS = [
  '#6366f1', '#8b5cf6', '#a855f7', '#ec4899', '#f43f5e',
  '#f97316', '#eab308', '#22c55e', '#14b8a6', '#06b6d4',
  '#3b82f6', '#6366f1', '#8b5cf6', '#a855f7', '#ec4899',
]

export default function ClassDistribution({ distribution, loading, selectedClass, onClassClick }: Props) {
  const totalAnnotations = distribution.reduce((s, d) => s + d.count, 0)
  const hasSelection = Boolean(selectedClass)

  const handleBarClick = (data: ClassCount) => {
    if (!onClassClick) return
    onClassClick(selectedClass === data.class_name ? null : data.class_name)
  }

  return (
    <div className="bg-slate-900 rounded-xl border border-slate-800 p-5 flex flex-col">
      {/* Header — fixed height */}
      <div className="flex items-center justify-between mb-4 flex-shrink-0">
        <div>
          <h3 className="text-sm font-semibold text-slate-200 uppercase tracking-wider">
            Class Distribution
          </h3>
          <p className="text-xs text-slate-500 mt-0.5">
            {distribution.length} classes · {totalAnnotations.toLocaleString()} total annotations
          </p>
        </div>
        {hasSelection && (
          <button
            onClick={() => onClassClick?.(null)}
            className="text-xs text-slate-400 hover:text-white px-2 py-1 rounded border border-slate-700 hover:border-slate-500 transition-colors"
          >
            Clear filter
          </button>
        )}
      </div>

      {/* Chart area — flex-1 so it fills the remaining panel height */}
      {loading ? (
        <div className="flex-1 bg-slate-800 rounded-lg animate-pulse" />
      ) : distribution.length === 0 ? (
        <div className="flex-1 flex items-center justify-center text-slate-600 text-sm">
          No data — prepare the dataset first
        </div>
      ) : (
        <>
          <div className="flex-1 min-h-0">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={distribution}
                layout="vertical"
                margin={{ top: 0, right: 16, left: 4, bottom: 0 }}
              >
                <XAxis
                  type="number"
                  tick={{ fill: '#64748b', fontSize: 11 }}
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
                  contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8, fontSize: 12 }}
                  labelStyle={{ color: '#f1f5f9' }}
                  itemStyle={{ color: '#94a3b8' }}
                  formatter={(v: number) => [v.toLocaleString(), 'annotations']}
                />
                <Bar
                  dataKey="count"
                  radius={[0, 4, 4, 0]}
                  maxBarSize={14}
                  onClick={handleBarClick}
                  style={{ cursor: onClassClick ? 'pointer' : 'default' }}
                >
                  {distribution.map((item, i) => (
                    <Cell
                      key={i}
                      fill={COLORS[i % COLORS.length]}
                      fillOpacity={hasSelection && item.class_name !== selectedClass ? 0.25 : 0.85}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Stats — fixed height at bottom */}
          <div className="grid grid-cols-3 gap-2 pt-3 mt-3 border-t border-slate-800 flex-shrink-0">
            <MiniStat label="Total classes" value={distribution.length} />
            <MiniStat label="Max count" value={distribution[0]?.count ?? 0} />
            <MiniStat label="Min count" value={distribution[distribution.length - 1]?.count ?? 0} />
          </div>
        </>
      )}
    </div>
  )
}

function MiniStat({ label, value }: { label: string; value: number }) {
  return (
    <div className="text-center">
      <div className="text-sm font-semibold text-white">{value.toLocaleString()}</div>
      <div className="text-xs text-slate-500">{label}</div>
    </div>
  )
}
