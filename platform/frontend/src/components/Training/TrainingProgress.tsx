import type { EpochResult, TrainingStatus } from '../../types'

interface Props {
  status: TrainingStatus | null
  epochResults: EpochResult[]
  onStop: () => void
  onComplete: () => void
  onReconfigure: () => void
}

export default function TrainingProgress({ status, epochResults, onStop, onComplete, onReconfigure }: Props) {
  const isRunning = status?.status === 'running'
  const isComplete = status?.status === 'completed'
  const isFailed = status?.status === 'failed'

  const totalEpochs = status?.configured_epochs ?? 0
  const completedEpochs = epochResults.length > 0
    ? epochResults[epochResults.length - 1].epoch + 1
    : 0
  const progress = totalEpochs > 0 ? Math.min(100, (completedEpochs / totalEpochs) * 100) : 0
  const elapsed = status?.elapsed_seconds ? formatDuration(status.elapsed_seconds) : null
  const latestEpoch = epochResults.length > 0 ? epochResults[epochResults.length - 1] : null

  return (
    <div className="bg-slate-900 rounded-xl border border-slate-800 p-5 flex flex-col h-full gap-4">
      <h3 className="text-sm font-semibold text-slate-200 uppercase tracking-wider flex-shrink-0">
        Training Progress
      </h3>

      {/* Status row */}
      <div className="flex items-center justify-between flex-shrink-0">
        <div className="flex items-center gap-3">
          <StatusDot status={status?.status ?? 'idle'} />
          <div>
            <div className="text-sm font-medium text-white">
              {isRunning ? 'Training in progress' :
               isComplete ? 'Training complete' :
               isFailed ? 'Training failed' : 'Idle'}
            </div>
            {elapsed && <div className="text-xs text-slate-400">Elapsed: {elapsed}</div>}
            {isFailed && status?.error && (
              <div className="text-xs text-red-400 mt-0.5">{status.error}</div>
            )}
          </div>
        </div>

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

      {/* MLflow run ID */}
      {status?.run_id && (
        <div className="flex-shrink-0">
          <span className="text-xs text-slate-500">MLflow run: </span>
          <code className="text-xs text-slate-300">{status.run_id.slice(0, 16)}…</code>
        </div>
      )}

      {/* Epoch progress bar */}
      {(isRunning || isComplete) && totalEpochs > 0 && (
        <div className="space-y-1 flex-shrink-0">
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

      {/* Latest metrics — flex-1 fills remaining space */}
      <div className="flex-1">
        {latestEpoch ? (
          <div className="space-y-3">
            <div className="text-xs text-slate-500 uppercase tracking-wider">
              Latest metrics — epoch {completedEpochs}
            </div>
            <div className="grid grid-cols-2 gap-2">
              <MetricTile label="mAP50"     value={latestEpoch.map50} />
              <MetricTile label="mAP50-95"  value={latestEpoch.map50_95} />
              <MetricTile label="Precision" value={latestEpoch.precision} />
              <MetricTile label="Recall"    value={latestEpoch.recall} />
            </div>
            <div className="grid grid-cols-2 gap-2">
              <LossTile label="Train box loss" value={latestEpoch.train_box_loss} />
              <LossTile label="Val box loss"   value={latestEpoch.val_box_loss} />
              <LossTile label="Train cls loss" value={latestEpoch.train_cls_loss} />
              <LossTile label="Val cls loss"   value={latestEpoch.val_cls_loss} />
            </div>
          </div>
        ) : (isRunning || isComplete) ? (
          <div className="text-xs text-slate-500">Waiting for first epoch…</div>
        ) : null}
      </div>

      {/* Action buttons */}
      {isComplete && (
        <button
          onClick={onComplete}
          className="w-full py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-medium rounded-lg transition-colors flex items-center justify-center gap-2 flex-shrink-0"
        >
          View Evaluation
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
        </button>
      )}
      {isFailed && (
        <button
          onClick={onReconfigure}
          className="w-full py-2.5 bg-slate-700 hover:bg-slate-600 text-white text-sm font-medium rounded-lg transition-colors flex items-center justify-center gap-2 flex-shrink-0"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
          Back to Configuration
        </button>
      )}
    </div>
  )
}

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds}s`
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`
  return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`
}

function StatusDot({ status }: { status: string }) {
  const color =
    status === 'running'   ? 'bg-amber-400 animate-pulse' :
    status === 'completed' ? 'bg-green-400' :
    status === 'failed'    ? 'bg-red-400' :
    'bg-slate-600'
  return <div className={`w-2.5 h-2.5 rounded-full flex-shrink-0 ${color}`} />
}

function MetricTile({ label, value }: { label: string; value: number }) {
  return (
    <div className="bg-slate-800 rounded-lg px-3 py-2 text-center">
      <div className="text-sm font-bold text-white">{(value * 100).toFixed(1)}%</div>
      <div className="text-xs text-slate-500">{label}</div>
    </div>
  )
}

function LossTile({ label, value }: { label: string; value: number }) {
  return (
    <div className="bg-slate-800/60 rounded-lg px-3 py-2 text-center">
      <div className="text-sm font-medium text-white">{value.toFixed(4)}</div>
      <div className="text-xs text-slate-500">{label}</div>
    </div>
  )
}
