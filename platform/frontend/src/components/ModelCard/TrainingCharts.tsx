import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts'
import type { EpochResult } from '../../types'

interface Props {
  epochResults: EpochResult[]
  loading: boolean
}

export default function TrainingCharts({ epochResults, loading }: Props) {
  if (loading) {
    return (
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-slate-900 rounded-xl border border-slate-800 p-5 h-64 animate-pulse" />
        <div className="bg-slate-900 rounded-xl border border-slate-800 p-5 h-64 animate-pulse" />
      </div>
    )
  }

  if (epochResults.length === 0) {
    return (
      <div className="bg-slate-900 rounded-xl border border-slate-800 p-5 text-center text-slate-500 text-sm py-12">
        No training history available. Complete a training run first.
      </div>
    )
  }

  const tooltipStyle = {
    contentStyle: { background: '#1e293b', border: '1px solid #334155', borderRadius: 8, fontSize: 11 },
    labelStyle: { color: '#f1f5f9' },
    labelFormatter: (v: number) => `Epoch ${v}`,
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Loss chart */}
      <div className="bg-slate-900 rounded-xl border border-slate-800 p-5 space-y-3">
        <div>
          <div className="text-sm font-medium text-slate-200">Training &amp; Validation Loss</div>
          <div className="text-xs text-slate-500">Box loss and classification loss per epoch</div>
        </div>
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={epochResults} margin={{ top: 4, right: 16, left: -8, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis dataKey="epoch" tick={{ fill: '#64748b', fontSize: 10 }}
              label={{ value: 'Epoch', position: 'insideBottomRight', offset: -4, fill: '#64748b', fontSize: 10 }} />
            <YAxis tick={{ fill: '#64748b', fontSize: 10 }} />
            <Tooltip {...tooltipStyle} formatter={(v: number) => [v.toFixed(4)]} />
            <Legend wrapperStyle={{ fontSize: 11, color: '#94a3b8' }} />
            <Line type="monotone" dataKey="train_box_loss" name="train box" stroke="#f97316" dot={false} strokeWidth={2} />
            <Line type="monotone" dataKey="val_box_loss" name="val box" stroke="#fb923c" dot={false} strokeWidth={1.5} strokeDasharray="4 2" />
            <Line type="monotone" dataKey="train_cls_loss" name="train cls" stroke="#a855f7" dot={false} strokeWidth={2} />
            <Line type="monotone" dataKey="val_cls_loss" name="val cls" stroke="#c084fc" dot={false} strokeWidth={1.5} strokeDasharray="4 2" />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Detection metrics chart */}
      <div className="bg-slate-900 rounded-xl border border-slate-800 p-5 space-y-3">
        <div>
          <div className="text-sm font-medium text-slate-200">Detection Metrics</div>
          <div className="text-xs text-slate-500">Precision · Recall · mAP50 · mAP50-95 per epoch</div>
        </div>
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={epochResults} margin={{ top: 4, right: 16, left: -8, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis dataKey="epoch" tick={{ fill: '#64748b', fontSize: 10 }}
              label={{ value: 'Epoch', position: 'insideBottomRight', offset: -4, fill: '#64748b', fontSize: 10 }} />
            <YAxis domain={[0, 1]} tick={{ fill: '#64748b', fontSize: 10 }} />
            <Tooltip {...tooltipStyle} formatter={(v: number) => [(v * 100).toFixed(1) + '%']} />
            <Legend wrapperStyle={{ fontSize: 11, color: '#94a3b8' }} />
            <Line type="monotone" dataKey="precision" stroke="#22c55e" dot={false} strokeWidth={2} />
            <Line type="monotone" dataKey="recall" stroke="#06b6d4" dot={false} strokeWidth={2} />
            <Line type="monotone" dataKey="map50" name="mAP50" stroke="#6366f1" dot={false} strokeWidth={2} />
            <Line type="monotone" dataKey="map50_95" name="mAP50-95" stroke="#8b5cf6" dot={false} strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
