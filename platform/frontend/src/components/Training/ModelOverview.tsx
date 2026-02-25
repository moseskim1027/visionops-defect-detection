import type { ModelInfo } from '../../types'

interface Props {
  info: ModelInfo | null
  loading: boolean
}

export default function ModelOverview({ info, loading }: Props) {
  if (loading) {
    return <div className="bg-slate-900 rounded-xl border border-slate-800 p-5 h-64 animate-pulse" />
  }

  const specs = info
    ? [
        { label: 'Parameters', value: info.parameters },
        { label: 'GFLOPs', value: info.gflops },
        { label: 'Input size', value: info.input_size },
        { label: 'Task', value: info.task },
        { label: 'Framework', value: info.framework },
      ]
    : []

  return (
    <div className="bg-slate-900 rounded-xl border border-slate-800 p-5 space-y-4">
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 bg-indigo-600/20 border border-indigo-600/30 rounded-lg flex items-center justify-center">
          <svg className="w-5 h-5 text-indigo-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
              d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
          </svg>
        </div>
        <div>
          <h3 className="text-white font-semibold">{info?.name ?? 'YOLOv8n'}</h3>
          <p className="text-xs text-slate-400">{info?.full_name ?? 'You Only Look Once v8 Nano'}</p>
        </div>
      </div>

      <p className="text-sm text-slate-400 leading-relaxed">{info?.description}</p>

      {/* Spec grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
        {specs.map(s => (
          <div key={s.label} className="bg-slate-800 rounded-lg px-3 py-2">
            <div className="text-xs text-slate-500">{s.label}</div>
            <div className="text-sm text-white font-medium mt-0.5">{s.value}</div>
          </div>
        ))}
      </div>

      {/* Architecture */}
      {info?.architecture && (
        <div className="bg-slate-800/50 rounded-lg px-3 py-2">
          <span className="text-xs text-slate-500">Architecture: </span>
          <span className="text-xs text-slate-300">{info.architecture}</span>
        </div>
      )}

      {/* Strengths */}
      {info?.strengths && (
        <div className="space-y-1.5">
          <div className="text-xs text-slate-500 uppercase tracking-wider">Key Strengths</div>
          <div className="flex flex-wrap gap-2">
            {info.strengths.map(s => (
              <span key={s} className="text-xs bg-indigo-950/60 text-indigo-300 border border-indigo-800/40 px-2 py-0.5 rounded-full">
                {s}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
