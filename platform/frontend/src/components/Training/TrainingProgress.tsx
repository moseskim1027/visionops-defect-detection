import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts'
import type { EpochResult, MLflowMetrics, TrainingStatus } from '../../types'

interface Props {
  status: TrainingStatus | null
  epochResults: EpochResult[]
  metrics: MLflowMetrics | null
  grafanaUrl: string
  mlflowUrl: string
  onStop: () => void
}

export default function TrainingProgress({
  status, epochResults, metrics, grafanaUrl, mlflowUrl, onStop,
}: Props) {
  const isRunning = status?.status === 'running'
  const isComplete = status?.status === 'completed'
  const isFailed = status?.status === 'failed'

  // Epoch progress — prefer configured_epochs from backend state; fall back to
  // MLflow params (available only after run is discovered) or the results length.
  const totalEpochs =
    status?.configured_epochs ??
    parseInt(metrics?.params?.epochs ?? '0') ??
    0

  const completedEpochs = epochResults.length > 0
    ? epochResults[epochResults.length - 1].epoch + 1  // epochs are 0-indexed in results.csv
    : 0

  const progress = totalEpochs > 0 ? Math.min(100, (completedEpochs / totalEpochs) * 100) : 0

  const elapsed = status?.elapsed_seconds ? formatDuration(status.elapsed_seconds) : null

  // Latest epoch metrics for the summary row
  const latestEpoch = epochResults.length > 0 ? epochResults[epochResults.length - 1] : null

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
                 isFailed ? (status?.error ?? 'Training failed') : 'Idle'}
              </div>
              {elapsed && (
                <div className="text-xs text-slate-400">Elapsed: {elapsed}</div>
              )}
            </div>
          </div>

          <div className="flex items-center gap-3">
            {status?.run_id && (
              <div className="text-right">
                <div className="text-xs text-slate-500">MLflow Run</div>
                <code className="text-xs text-slate-300">{status.run_id.slice(0, 12)}…</code>
              </div>
            )}
            {isRunning && (
              <button
                onClick={onStop}
                className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-red-400 border border-red-800/60 hover:bg-red-950/40 rounded-lg transition-colors"
              >
                <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 24 24">
                  <rect x="6" y="6" width="12" height="12" rx="1" />
                </svg>
                Stop
              </button>
            )}
          </div>
        </div>

        {/* Epoch progress bar */}
        {(isRunning || isComplete) && totalEpochs > 0 && (
          <div className="space-y-1">
            <div className="flex justify-between text-xs text-slate-400">
              <span>Epochs</span>
              <span>{completedEpochs} / {totalEpochs}</span>
            </div>
            <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-indigo-500 rounded-full transition-all duration-700"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>
        )}

        {/* Latest metrics summary */}
        {latestEpoch && (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-2 border-t border-slate-800">
            <MetricBadge label="mAP50" value={latestEpoch.map50} />
            <MetricBadge label="mAP50-95" value={latestEpoch.map50_95} />
            <MetricBadge label="Precision" value={latestEpoch.precision} />
            <MetricBadge label="Recall" value={latestEpoch.recall} />
          </div>
        )}
      </div>

      {/* Training loss chart */}
      {epochResults.length > 0 && (
        <ChartCard
          title="Training & Validation Loss"
          subtitle="Box loss and classification loss per epoch (from results.csv)"
        >
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={epochResults} margin={{ top: 4, right: 16, left: -8, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis
                dataKey="epoch"
                tick={{ fill: '#64748b', fontSize: 10 }}
                label={{ value: 'Epoch', position: 'insideBottomRight', offset: -4, fill: '#64748b', fontSize: 10 }}
              />
              <YAxis tick={{ fill: '#64748b', fontSize: 10 }} />
              <Tooltip
                contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8, fontSize: 11 }}
                labelStyle={{ color: '#f1f5f9' }}
                labelFormatter={(v) => `Epoch ${v}`}
                formatter={(v: number) => [v.toFixed(4)]}
              />
              <Legend wrapperStyle={{ fontSize: 11, color: '#94a3b8' }} />
              <Line type="monotone" dataKey="train_box_loss" name="train box" stroke="#f97316" dot={false} strokeWidth={2} />
              <Line type="monotone" dataKey="val_box_loss" name="val box" stroke="#fb923c" dot={false} strokeWidth={1.5} strokeDasharray="4 2" />
              <Line type="monotone" dataKey="train_cls_loss" name="train cls" stroke="#a855f7" dot={false} strokeWidth={2} />
              <Line type="monotone" dataKey="val_cls_loss" name="val cls" stroke="#c084fc" dot={false} strokeWidth={1.5} strokeDasharray="4 2" />
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>
      )}

      {/* Detection metrics chart */}
      {epochResults.length > 0 && (
        <ChartCard
          title="Detection Metrics"
          subtitle="Precision · Recall · mAP50 · mAP50-95 per epoch"
        >
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={epochResults} margin={{ top: 4, right: 16, left: -8, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis
                dataKey="epoch"
                tick={{ fill: '#64748b', fontSize: 10 }}
                label={{ value: 'Epoch', position: 'insideBottomRight', offset: -4, fill: '#64748b', fontSize: 10 }}
              />
              <YAxis domain={[0, 1]} tick={{ fill: '#64748b', fontSize: 10 }} />
              <Tooltip
                contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8, fontSize: 11 }}
                labelStyle={{ color: '#f1f5f9' }}
                labelFormatter={(v) => `Epoch ${v}`}
                formatter={(v: number) => [(v * 100).toFixed(1) + '%']}
              />
              <Legend wrapperStyle={{ fontSize: 11, color: '#94a3b8' }} />
              <Line type="monotone" dataKey="precision" stroke="#22c55e" dot={false} strokeWidth={2} />
              <Line type="monotone" dataKey="recall" stroke="#06b6d4" dot={false} strokeWidth={2} />
              <Line type="monotone" dataKey="map50" name="mAP50" stroke="#6366f1" dot={false} strokeWidth={2} />
              <Line type="monotone" dataKey="map50_95" name="mAP50-95" stroke="#8b5cf6" dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>
      )}

      {/* External service links */}
      <div className="grid grid-cols-2 gap-4">
        <ExternalCard
          title="MLflow Experiment Tracker"
          description="Full run history, artifacts, and model registry"
          href={mlflowUrl}
          color="indigo"
        />
        <ExternalCard
          title="Grafana Monitoring"
          description="Real-time inference metrics — latency, throughput, errors"
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
  return <div className={`w-2.5 h-2.5 rounded-full flex-shrink-0 ${color}`} />
}

function MetricBadge({ label, value }: { label: string; value: number }) {
  return (
    <div className="text-center">
      <div className="text-sm font-bold text-white">{(value * 100).toFixed(1)}%</div>
      <div className="text-xs text-slate-500">{label}</div>
    </div>
  )
}

function ChartCard({ title, subtitle, children }: {
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

function ExternalCard({ title, description, href, color }: {
  title: string; description: string; href: string; color: 'indigo' | 'orange'
}) {
  const cls = color === 'indigo'
    ? 'border-indigo-800/40 bg-indigo-950/30 text-indigo-300 hover:bg-indigo-950/50'
    : 'border-orange-800/40 bg-orange-950/30 text-orange-300 hover:bg-orange-950/50'
  return (
    <a href={href} target="_blank" rel="noreferrer"
      className={`block rounded-xl border p-4 transition-colors ${cls}`}
    >
      <div className="flex items-start justify-between">
        <div>
          <div className="text-sm font-medium">{title}</div>
          <div className="text-xs opacity-70 mt-1">{description}</div>
        </div>
        <svg className="w-4 h-4 mt-0.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
            d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
        </svg>
      </div>
    </a>
  )
}
