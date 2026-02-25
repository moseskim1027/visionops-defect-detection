import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts'
import type { MLflowMetrics, TrainingStatus } from '../../types'

interface Props {
  status: TrainingStatus | null
  metrics: MLflowMetrics | null
  grafanaUrl: string
  mlflowUrl: string
}

export default function TrainingProgress({ status, metrics, grafanaUrl, mlflowUrl }: Props) {
  const isRunning = status?.status === 'running'
  const isComplete = status?.status === 'completed'
  const isFailed = status?.status === 'failed'

  const epochs = parseInt(metrics?.params?.epochs ?? '0')
  const currentStep = getMaxStep(metrics?.metrics ?? {})
  const progress = epochs > 0 ? Math.min(100, (currentStep / epochs) * 100) : 0

  // Build epoch-aligned chart data
  const lossData = buildChartData(metrics?.metrics ?? {}, [
    'train_box_loss', 'val_box_loss', 'train_cls_loss', 'val_cls_loss',
  ])
  const metricsData = buildChartData(metrics?.metrics ?? {}, [
    'precision', 'recall', 'map50', 'map50_95',
  ])

  const elapsed = status?.elapsed_seconds
    ? formatDuration(status.elapsed_seconds)
    : null

  return (
    <div className="space-y-5">
      {/* Status bar */}
      <div className="bg-slate-900 rounded-xl border border-slate-800 p-5 space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <StatusDot status={status?.status ?? 'idle'} />
            <div>
              <div className="text-sm font-medium text-white">
                {isRunning ? 'Training in progress' :
                 isComplete ? 'Training complete' :
                 isFailed ? 'Training failed' : 'Ready to train'}
              </div>
              {elapsed && (
                <div className="text-xs text-slate-400">Elapsed: {elapsed}</div>
              )}
            </div>
          </div>
          {status?.run_id && (
            <div className="text-right">
              <div className="text-xs text-slate-500">MLflow Run</div>
              <code className="text-xs text-slate-300">{status.run_id.slice(0, 12)}…</code>
            </div>
          )}
        </div>

        {/* Progress bar */}
        {(isRunning || isComplete) && epochs > 0 && (
          <div className="space-y-1">
            <div className="flex justify-between text-xs text-slate-400">
              <span>Epoch progress</span>
              <span>{currentStep} / {epochs}</span>
            </div>
            <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-indigo-500 rounded-full transition-all duration-500"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>
        )}

        {/* Error */}
        {isFailed && status?.error && (
          <div className="bg-red-950/40 border border-red-800/50 rounded-lg p-3 text-xs text-red-400 font-mono overflow-auto max-h-32">
            {status.error}
          </div>
        )}

        {/* Current final metrics */}
        {metrics?.metrics && Object.keys(metrics.metrics).length > 0 && (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-2 border-t border-slate-800">
            {['map50', 'map50_95', 'precision', 'recall'].map(key => {
              const pts = metrics.metrics[key]
              const last = pts?.[pts.length - 1]?.value
              return last !== undefined ? (
                <MetricBadge key={key} label={key} value={last} />
              ) : null
            })}
          </div>
        )}
      </div>

      {/* Loss chart */}
      {lossData.length > 0 && (
        <ChartCard title="Training & Validation Loss" subtitle="Box loss + classification loss per epoch">
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={lossData} margin={{ top: 4, right: 16, left: -8, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="step" tick={{ fill: '#64748b', fontSize: 10 }} />
              <YAxis tick={{ fill: '#64748b', fontSize: 10 }} />
              <Tooltip
                contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8, fontSize: 11 }}
                labelStyle={{ color: '#f1f5f9' }}
                itemStyle={{ color: '#94a3b8' }}
              />
              <Legend wrapperStyle={{ fontSize: 11, color: '#94a3b8' }} />
              <Line type="monotone" dataKey="train_box_loss" stroke="#f97316" dot={false} strokeWidth={2} />
              <Line type="monotone" dataKey="val_box_loss" stroke="#fb923c" dot={false} strokeWidth={1.5} strokeDasharray="4 2" />
              <Line type="monotone" dataKey="train_cls_loss" stroke="#a855f7" dot={false} strokeWidth={2} />
              <Line type="monotone" dataKey="val_cls_loss" stroke="#c084fc" dot={false} strokeWidth={1.5} strokeDasharray="4 2" />
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>
      )}

      {/* Metrics chart */}
      {metricsData.length > 0 && (
        <ChartCard title="Detection Metrics" subtitle="Precision · Recall · mAP50 · mAP50-95 per epoch">
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={metricsData} margin={{ top: 4, right: 16, left: -8, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="step" tick={{ fill: '#64748b', fontSize: 10 }} />
              <YAxis domain={[0, 1]} tick={{ fill: '#64748b', fontSize: 10 }} />
              <Tooltip
                contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8, fontSize: 11 }}
                labelStyle={{ color: '#f1f5f9' }}
                itemStyle={{ color: '#94a3b8' }}
              />
              <Legend wrapperStyle={{ fontSize: 11, color: '#94a3b8' }} />
              <Line type="monotone" dataKey="precision" stroke="#22c55e" dot={false} strokeWidth={2} />
              <Line type="monotone" dataKey="recall" stroke="#06b6d4" dot={false} strokeWidth={2} />
              <Line type="monotone" dataKey="map50" stroke="#6366f1" dot={false} strokeWidth={2} />
              <Line type="monotone" dataKey="map50_95" stroke="#8b5cf6" dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>
      )}

      {/* External service links */}
      <div className="grid grid-cols-2 gap-4">
        <ExternalCard
          title="MLflow Experiment Tracker"
          description="View full experiment history, artifacts, and model registry"
          href={mlflowUrl}
          color="indigo"
        />
        <ExternalCard
          title="Grafana Monitoring"
          description="Real-time inference metrics — latency, throughput, error rate"
          href={grafanaUrl}
          color="orange"
        />
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function getMaxStep(metrics: Record<string, { step: number }[]>): number {
  let max = 0
  for (const pts of Object.values(metrics)) {
    if (pts.length > 0) max = Math.max(max, pts[pts.length - 1].step)
  }
  return max
}

function buildChartData(
  metrics: Record<string, { step: number; value: number }[]>,
  keys: string[],
): Record<string, number>[] {
  const stepSet = new Set<number>()
  for (const k of keys) {
    (metrics[k] ?? []).forEach(p => stepSet.add(p.step))
  }
  const steps = Array.from(stepSet).sort((a, b) => a - b)
  return steps.map(step => {
    const row: Record<string, number> = { step }
    for (const k of keys) {
      const found = (metrics[k] ?? []).find(p => p.step === step)
      if (found) row[k] = parseFloat(found.value.toFixed(4))
    }
    return row
  })
}

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds}s`
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`
  return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`
}

function StatusDot({ status }: { status: string }) {
  const color =
    status === 'running' ? 'bg-amber-400 animate-pulse' :
    status === 'completed' ? 'bg-green-400' :
    status === 'failed' ? 'bg-red-400' :
    'bg-slate-600'
  return <div className={`w-2.5 h-2.5 rounded-full ${color}`} />
}

function MetricBadge({ label, value }: { label: string; value: number }) {
  return (
    <div className="text-center">
      <div className="text-sm font-bold text-white">{(value * 100).toFixed(1)}%</div>
      <div className="text-xs text-slate-500">{label}</div>
    </div>
  )
}

function ChartCard({
  title, subtitle, children,
}: {
  title: string; subtitle: string; children: React.ReactNode
}) {
  return (
    <div className="bg-slate-900 rounded-xl border border-slate-800 p-5 space-y-3">
      <div>
        <div className="text-sm font-medium text-slate-200">{title}</div>
        <div className="text-xs text-slate-500">{subtitle}</div>
      </div>
      {children}
    </div>
  )
}

function ExternalCard({
  title, description, href, color,
}: {
  title: string; description: string; href: string; color: 'indigo' | 'orange'
}) {
  const accent = color === 'indigo' ? 'border-indigo-800/40 bg-indigo-950/30 text-indigo-300 hover:bg-indigo-950/50' : 'border-orange-800/40 bg-orange-950/30 text-orange-300 hover:bg-orange-950/50'
  return (
    <a
      href={href}
      target="_blank"
      rel="noreferrer"
      className={`block rounded-xl border p-4 transition-colors ${accent}`}
    >
      <div className="flex items-start justify-between">
        <div>
          <div className="text-sm font-medium">{title}</div>
          <div className="text-xs opacity-70 mt-1">{description}</div>
        </div>
        <svg className="w-4 h-4 mt-0.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
        </svg>
      </div>
    </a>
  )
}
