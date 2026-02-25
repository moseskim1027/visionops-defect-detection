import { useCallback, useEffect, useState } from 'react'
import { api } from '../../api/client'
import type { InferenceStatus, ModelVersion } from '../../types'

// ---------------------------------------------------------------------------
// Model Registry panel
// ---------------------------------------------------------------------------

function VersionRow({
  v,
  promoting,
  onPromote,
}: {
  v: ModelVersion
  promoting: boolean
  onPromote: (version: string) => void
}) {
  const isProduction = v.aliases.includes('production')
  const date = v.creation_timestamp
    ? new Date(v.creation_timestamp).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })
    : '—'

  return (
    <div className={`flex items-center gap-4 px-4 py-3 rounded-lg border transition-colors ${
      isProduction ? 'bg-green-950/20 border-green-800/40' : 'bg-slate-800/40 border-slate-700/50'
    }`}>
      {/* Version badge */}
      <div className="flex-shrink-0">
        <span className="text-sm font-mono font-bold text-slate-200">v{v.version}</span>
      </div>

      {/* Aliases + status */}
      <div className="flex items-center gap-2 flex-shrink-0">
        {isProduction && (
          <span className="text-xs bg-green-900/60 text-green-300 border border-green-700/50 px-2 py-0.5 rounded-full font-medium">
            production
          </span>
        )}
        {v.aliases.filter(a => a !== 'production').map(a => (
          <span key={a} className="text-xs bg-slate-700/60 text-slate-300 border border-slate-600/50 px-2 py-0.5 rounded-full">
            {a}
          </span>
        ))}
      </div>

      {/* Metrics */}
      <div className="flex items-center gap-3 flex-1 min-w-0">
        {v.metrics.map50 !== undefined && (
          <span className="text-xs text-slate-400">
            mAP <span className="text-slate-200 font-medium">{(v.metrics.map50 * 100).toFixed(1)}%</span>
          </span>
        )}
        {v.metrics.precision !== undefined && (
          <span className="text-xs text-slate-400 hidden sm:inline">
            P <span className="text-slate-200 font-medium">{(v.metrics.precision * 100).toFixed(1)}%</span>
          </span>
        )}
        {v.metrics.recall !== undefined && (
          <span className="text-xs text-slate-400 hidden sm:inline">
            R <span className="text-slate-200 font-medium">{(v.metrics.recall * 100).toFixed(1)}%</span>
          </span>
        )}
        <span className="text-xs text-slate-600 ml-auto flex-shrink-0">{date}</span>
      </div>

      {/* Promote button */}
      <button
        onClick={() => onPromote(v.version)}
        disabled={isProduction || promoting}
        className={`flex-shrink-0 px-3 py-1.5 text-xs font-medium rounded-lg transition-colors ${
          isProduction
            ? 'bg-slate-800 text-slate-600 cursor-not-allowed border border-slate-700'
            : promoting
            ? 'bg-slate-700 text-slate-400 cursor-wait border border-slate-600'
            : 'bg-indigo-700 hover:bg-indigo-600 text-white border border-indigo-600'
        }`}
      >
        {isProduction ? 'Active' : promoting ? 'Promoting…' : 'Set Production'}
      </button>

      {/* MLflow run link — far right */}
      {v.experiment_id && v.run_id && (
        <a
          href={`http://localhost:5001/#/experiments/${v.experiment_id}/runs/${v.run_id}`}
          target="_blank"
          rel="noreferrer"
          title="View MLflow run"
          className="flex-shrink-0 p-1.5 text-slate-500 hover:text-[#43C9ED] rounded-md transition-colors"
        >
          <svg viewBox="0 0 24 24" className="w-4 h-4" fill="currentColor">
            <path d="M11.883.002a12.044 12.044 0 0 0-9.326 19.463l3.668-2.694A7.573 7.573 0 0 1 12.043 4.45v2.867l6.908-5.14A12 12 0 0 0 11.883.002m9.562 4.533L17.777 7.23a7.573 7.573 0 0 1-5.818 12.322v-2.867l-6.908 5.14a12.046 12.046 0 0 0 16.394-17.29"/>
          </svg>
        </a>
      )}
    </div>
  )
}

function ModelRegistryPanel() {
  const [versions, setVersions] = useState<ModelVersion[]>([])
  const [modelName, setModelName] = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [promotingVersion, setPromotingVersion] = useState<string | null>(null)
  const [promoteMsg, setPromoteMsg] = useState<{ type: 'ok' | 'err'; text: string } | null>(null)

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await api.getModelVersions()
      setVersions(res.versions ?? [])
      setModelName(res.model_name ?? '')
      if (res.error) setError(res.error)
    } catch (e: any) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { load() }, [load])

  const handlePromote = async (version: string) => {
    setPromotingVersion(version)
    setPromoteMsg(null)
    try {
      const res = await api.promoteModel(version)
      if (res.error) {
        setPromoteMsg({ type: 'err', text: res.error })
      } else {
        setPromoteMsg({ type: 'ok', text: `v${version} promoted to production` })
        await load()
      }
    } catch (e: any) {
      setPromoteMsg({ type: 'err', text: e.message })
    } finally {
      setPromotingVersion(null)
    }
  }

  return (
    <div className="bg-slate-900 rounded-xl border border-slate-800 p-5 flex flex-col gap-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-sm font-semibold text-slate-200 uppercase tracking-wider">
            Model Registry
          </h3>
          {modelName && (
            <p className="text-xs text-slate-500 mt-0.5 font-mono">{modelName}</p>
          )}
        </div>
        <button
          onClick={load}
          disabled={loading}
          className="p-1.5 text-slate-500 hover:text-slate-300 rounded-md transition-colors"
          title="Refresh"
        >
          <svg className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
              d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
        </button>
      </div>

      {/* Promote feedback */}
      {promoteMsg && (
        <div className={`text-xs px-3 py-2 rounded-lg border ${
          promoteMsg.type === 'ok'
            ? 'bg-green-950/40 border-green-800/50 text-green-300'
            : 'bg-red-950/40 border-red-800/50 text-red-300'
        }`}>
          {promoteMsg.text}
        </div>
      )}

      {/* Error */}
      {error && !loading && (
        <div className="text-xs bg-amber-950/40 border border-amber-800/50 rounded-lg p-3 text-amber-300">
          {error.includes('RESOURCE_DOES_NOT_EXIST') || error.includes('not found')
            ? 'No registered models found. Complete a training run first.'
            : error}
        </div>
      )}

      {/* Version list — scrollable past 5 rows */}
      <div className="flex flex-col gap-2 max-h-[22rem] overflow-y-auto pr-0.5">
        {loading ? (
          Array.from({ length: 5 }).map((_, i) => (
            <div key={i} className="h-14 bg-slate-800 rounded-lg animate-pulse flex-shrink-0" />
          ))
        ) : versions.length === 0 ? (
          <div className="py-8 text-center text-slate-500 text-sm">
            No model versions registered yet.
          </div>
        ) : (
          versions.map(v => (
            <VersionRow
              key={v.version}
              v={v}
              promoting={promotingVersion === v.version}
              onPromote={handlePromote}
            />
          ))
        )}
      </div>

      {/* Explanation */}
      <div className="flex items-start gap-2 text-xs text-slate-500 bg-slate-800/40 rounded-lg px-3 py-2.5">
        <svg className="w-3.5 h-3.5 flex-shrink-0 mt-0.5 text-slate-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        <span>
          <span className="text-slate-400 font-medium">Deploy Model</span> loads the version marked{' '}
          <span className="font-mono text-green-400">production</span> into the live inference service.
          Set a version as production above, then click Deploy Model on the right.
        </span>
      </div>

      {/* Divider */}
      <div className="border-t border-slate-800" />

      {/* MLflow link */}
      <a
        href="http://localhost:5001"
        target="_blank"
        rel="noreferrer"
        className="w-full py-2.5 text-sm font-medium rounded-lg border border-slate-700 hover:border-slate-500 text-slate-400 hover:text-white transition-colors flex items-center justify-center gap-2"
      >
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
            d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
        </svg>
        View MLflow Registry ↗
      </a>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Inference Service panel
// ---------------------------------------------------------------------------

function StatusDot({ status, modelLoaded }: { status: string; modelLoaded: boolean }) {
  if (status === 'unreachable') {
    return (
      <span className="flex items-center gap-2">
        <span className="w-2.5 h-2.5 rounded-full bg-red-500 flex-shrink-0" />
        <span className="text-red-400 text-sm font-medium">Unreachable</span>
      </span>
    )
  }
  if (!modelLoaded) {
    return (
      <span className="flex items-center gap-2">
        <span className="w-2.5 h-2.5 rounded-full bg-yellow-500 flex-shrink-0" />
        <span className="text-yellow-400 text-sm font-medium">Running — no model</span>
      </span>
    )
  }
  return (
    <span className="flex items-center gap-2">
      <span className="w-2.5 h-2.5 rounded-full bg-green-500 flex-shrink-0 shadow-[0_0_6px_rgba(34,197,94,0.7)]" />
      <span className="text-green-400 text-sm font-medium">Online</span>
    </span>
  )
}

type TestResult =
  | { kind: 'single'; image: string; source_dir: string; num_detections: number; inference_time_ms: number; detections: { class_name: string; confidence: number }[] }
  | { kind: 'batch'; source_dir: string; images_sent: number; errors: number; total_detections: number; avg_detections_per_image: number }
  | { kind: 'error'; message: string }

function TestResultCard({ result }: { result: TestResult }) {
  if (result.kind === 'error') {
    return (
      <div className="text-xs bg-red-950/40 border border-red-800/50 rounded-lg px-3 py-2.5 text-red-300">
        {result.message}
      </div>
    )
  }
  if (result.kind === 'single') {
    const classes = result.detections.map(d => d.class_name)
    const unique = [...new Set(classes)]
    return (
      <div className="text-xs bg-green-950/30 border border-green-800/40 rounded-lg px-3 py-2.5 space-y-1">
        <div className="flex items-center gap-2">
          <span className="text-green-400 font-medium">{result.num_detections} detection{result.num_detections !== 1 ? 's' : ''}</span>
          <span className="text-slate-500">·</span>
          <span className="text-slate-400 font-mono">{result.image}</span>
          <span className="text-slate-500 ml-auto">{result.inference_time_ms.toFixed(0)} ms</span>
        </div>
        {unique.length > 0 && (
          <div className="flex flex-wrap gap-1 pt-0.5">
            {unique.map(c => (
              <span key={c} className="bg-slate-700/60 text-slate-300 px-1.5 py-0.5 rounded text-[11px]">{c}</span>
            ))}
          </div>
        )}
      </div>
    )
  }
  // batch
  return (
    <div className={`text-xs rounded-lg px-3 py-2.5 border space-y-1 ${
      result.errors > 0
        ? 'bg-amber-950/30 border-amber-800/40'
        : 'bg-green-950/30 border-green-800/40'
    }`}>
      <div className="flex items-center gap-2">
        <span className={`font-medium ${result.errors > 0 ? 'text-amber-400' : 'text-green-400'}`}>
          {result.images_sent} images sent
        </span>
        {result.errors > 0 && <span className="text-red-400">{result.errors} errors</span>}
      </div>
      <div className="text-slate-400">
        {result.total_detections} total detections · {result.avg_detections_per_image} avg/image
        <span className="text-slate-600 ml-2 font-mono">{result.source_dir}</span>
      </div>
    </div>
  )
}

function InferencePanel() {
  const [status, setStatus] = useState<InferenceStatus | null>(null)
  const [statusLoading, setStatusLoading] = useState(true)
  const [reloading, setReloading] = useState(false)
  const [reloadMsg, setReloadMsg] = useState<{ type: 'ok' | 'err'; text: string } | null>(null)
  const [testing, setTesting] = useState<'single' | 'batch' | null>(null)
  const [testResult, setTestResult] = useState<TestResult | null>(null)

  const loadStatus = useCallback(async () => {
    setStatusLoading(true)
    try {
      const res = await api.getInferenceStatus()
      setStatus(res)
    } finally {
      setStatusLoading(false)
    }
  }, [])

  useEffect(() => { loadStatus() }, [loadStatus])

  const handleReload = async () => {
    setReloading(true)
    setReloadMsg(null)
    try {
      const res = await api.reloadInference()
      if (res.error || res.status === 'error') {
        setReloadMsg({ type: 'err', text: res.detail ?? res.error ?? 'Reload failed' })
      } else {
        const src = res.source === 'mlflow' ? `MLflow (alias: ${res.alias ?? 'production'})` : 'local weights'
        setReloadMsg({ type: 'ok', text: `Reloaded from ${src}` })
        await loadStatus()
      }
    } catch (e: any) {
      setReloadMsg({ type: 'err', text: e.message })
    } finally {
      setReloading(false)
    }
  }

  const handleTestSingle = async () => {
    setTesting('single')
    setTestResult(null)
    try {
      const res = await api.testPredict()
      if (res.error) {
        setTestResult({ kind: 'error', message: res.error })
      } else {
        setTestResult({ kind: 'single', ...res })
      }
    } catch (e: any) {
      setTestResult({ kind: 'error', message: e.message })
    } finally {
      setTesting(null)
    }
  }

  const handleTestBatch = async () => {
    setTesting('batch')
    setTestResult(null)
    try {
      const res = await api.testBatchPredict()
      if (res.error) {
        setTestResult({ kind: 'error', message: res.error })
      } else {
        setTestResult({ kind: 'batch', ...res })
      }
    } catch (e: any) {
      setTestResult({ kind: 'error', message: e.message })
    } finally {
      setTesting(null)
    }
  }

  const busy = reloading || testing !== null

  return (
    <div className="bg-slate-900 rounded-xl border border-slate-800 p-5 flex flex-col gap-5">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-slate-200 uppercase tracking-wider">
          Inference Service
        </h3>
        <button
          onClick={loadStatus}
          disabled={statusLoading}
          className="p-1.5 text-slate-500 hover:text-slate-300 rounded-md transition-colors"
          title="Refresh status"
        >
          <svg className={`w-4 h-4 ${statusLoading ? 'animate-spin' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
              d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
        </button>
      </div>

      {/* Status */}
      <div className="bg-slate-800/50 rounded-lg px-4 py-4 space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-xs text-slate-500 uppercase tracking-wider">Status</span>
          {statusLoading ? (
            <div className="h-4 w-20 bg-slate-700 rounded animate-pulse" />
          ) : status ? (
            <StatusDot status={status.status} modelLoaded={status.model_loaded} />
          ) : (
            <span className="text-slate-500 text-sm">—</span>
          )}
        </div>

        <div className="flex items-center justify-between">
          <span className="text-xs text-slate-500 uppercase tracking-wider">Model Loaded</span>
          {statusLoading ? (
            <div className="h-4 w-12 bg-slate-700 rounded animate-pulse" />
          ) : (
            <span className={`text-sm font-medium ${status?.model_loaded ? 'text-green-400' : 'text-slate-500'}`}>
              {status?.model_loaded ? 'Yes' : 'No'}
            </span>
          )}
        </div>

        {status?.error && status.status !== 'unreachable' && (
          <div className="text-xs text-red-400 bg-red-950/30 border border-red-900/40 rounded px-2 py-1.5">
            {status.error}
          </div>
        )}
      </div>

      {/* Reload feedback */}
      {reloadMsg && (
        <div className={`text-xs px-3 py-2 rounded-lg border ${
          reloadMsg.type === 'ok'
            ? 'bg-green-950/40 border-green-800/50 text-green-300'
            : 'bg-red-950/40 border-red-800/50 text-red-300'
        }`}>
          {reloadMsg.text}
        </div>
      )}

      {/* Reload button */}
      <button
        onClick={handleReload}
        disabled={busy}
        className={`w-full py-2.5 text-sm font-medium rounded-lg transition-colors flex items-center justify-center gap-2 ${
          reloading
            ? 'bg-slate-700 text-slate-400 cursor-wait'
            : busy
            ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
            : 'bg-indigo-700 hover:bg-indigo-600 text-white'
        }`}
      >
        {reloading ? (
          <>
            <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
            Deploying…
          </>
        ) : (
          <>
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
            </svg>
            Deploy Model
          </>
        )}
      </button>

      {/* Divider */}
      <div className="border-t border-slate-800" />

      {/* Test buttons */}
      <div className="flex flex-col gap-3">
        <p className="text-xs text-slate-500 uppercase tracking-wider">Test Endpoint</p>
        <div className="flex gap-2">
          <button
            onClick={handleTestSingle}
            disabled={busy}
            className={`flex-1 py-2 text-sm font-medium rounded-lg transition-colors flex items-center justify-center gap-2 ${
              testing === 'single'
                ? 'bg-slate-700 text-slate-400 cursor-wait'
                : busy
                ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
                : 'bg-slate-700 hover:bg-slate-600 text-slate-200'
            }`}
          >
            {testing === 'single' ? (
              <svg className="w-3.5 h-3.5 animate-spin" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                  d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
            ) : (
              <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            )}
            Test Single
          </button>
          <button
            onClick={handleTestBatch}
            disabled={busy}
            className={`flex-1 py-2 text-sm font-medium rounded-lg transition-colors flex items-center justify-center gap-2 ${
              testing === 'batch'
                ? 'bg-slate-700 text-slate-400 cursor-wait'
                : busy
                ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
                : 'bg-slate-700 hover:bg-slate-600 text-slate-200'
            }`}
          >
            {testing === 'batch' ? (
              <svg className="w-3.5 h-3.5 animate-spin" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                  d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
            ) : (
              <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
              </svg>
            )}
            Test Batch
          </button>
        </div>

        {testResult && <TestResultCard result={testResult} />}
      </div>

      {/* Divider */}
      <div className="border-t border-slate-800" />

      {/* Grafana link */}
      <a
        href="http://localhost:3000/d/visionops-inference/visionops-inference?orgId=1&from=now-5m&to=now&timezone=browser&var-datasource=prometheus&refresh=30s"
        target="_blank"
        rel="noreferrer"
        className="w-full py-2.5 text-sm font-medium rounded-lg border border-slate-700 hover:border-slate-500 text-slate-400 hover:text-white transition-colors flex items-center justify-center gap-2"
      >
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
            d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
        </svg>
        View Grafana Dashboard ↗
      </a>
    </div>
  )
}

// ---------------------------------------------------------------------------
// DeployStep
// ---------------------------------------------------------------------------

export default function DeployStep() {
  return (
    <div className="max-w-6xl mx-auto px-6 py-8 space-y-8">
      {/* Title */}
      <div>
        <h2 className="text-2xl font-bold text-white">Deploy</h2>
        <p className="text-slate-400 mt-1 text-sm">
          Promote a model version to production and reload the inference service.
        </p>
      </div>

      {/* Two-column layout */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <ModelRegistryPanel />
        <InferencePanel />
      </div>
    </div>
  )
}
