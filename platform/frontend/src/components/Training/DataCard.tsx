import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
} from 'recharts'
import type { ClassCount } from '../../types'

interface Props {
  distribution: ClassCount[]
  loading: boolean
}

const COLORS = [
  '#6366f1', '#8b5cf6', '#a855f7', '#ec4899', '#f43f5e',
  '#f97316', '#eab308', '#22c55e', '#14b8a6', '#06b6d4',
  '#3b82f6', '#6366f1', '#8b5cf6', '#a855f7', '#ec4899',
]

export default function DataCard({ distribution, loading }: Props) {
  const totalAnnotations = distribution.reduce((s, d) => s + d.count, 0)
  const topN = distribution.slice(0, 20)

  return (
    <div className="bg-slate-900 rounded-xl border border-slate-800 p-5 space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-sm font-semibold text-slate-200 uppercase tracking-wider">Data Card</h3>
          <p className="text-xs text-slate-500 mt-0.5">
            {distribution.length} classes · {totalAnnotations.toLocaleString()} total annotations
          </p>
        </div>
        <div className="text-right">
          <div className="text-xs text-slate-500">Distribution</div>
          <div className="text-xs text-slate-400">top {Math.min(20, distribution.length)} classes shown</div>
        </div>
      </div>

      {loading ? (
        <div className="h-56 bg-slate-800 rounded-lg animate-pulse" />
      ) : distribution.length === 0 ? (
        <div className="h-56 flex items-center justify-center text-slate-600 text-sm">
          No data — prepare the dataset first
        </div>
      ) : (
        <>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart
              data={topN}
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
                contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8, fontSize: 12 }}
                labelStyle={{ color: '#f1f5f9' }}
                itemStyle={{ color: '#94a3b8' }}
                formatter={(v: number) => [v.toLocaleString(), 'annotations']}
              />
              <Bar dataKey="count" radius={[0, 4, 4, 0]} maxBarSize={14}>
                {topN.map((_, i) => (
                  <Cell key={i} fill={COLORS[i % COLORS.length]} fillOpacity={0.85} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>

          {/* Summary stats */}
          <div className="grid grid-cols-3 gap-2 pt-2 border-t border-slate-800">
            <MiniStat label="Total classes" value={distribution.length} />
            <MiniStat label="Max count" value={distribution[0]?.count ?? 0} />
            <MiniStat
              label="Min count"
              value={distribution[distribution.length - 1]?.count ?? 0}
            />
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
