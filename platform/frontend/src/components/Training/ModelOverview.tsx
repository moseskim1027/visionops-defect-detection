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
        { label: 'Model',      value: info.name },
        { label: 'Parameters', value: info.parameters },
        { label: 'GFLOPs',     value: info.gflops },
        { label: 'Input size', value: info.input_size },
        { label: 'Task',       value: info.task },
        { label: 'Framework',  value: info.framework },
      ]
    : []

  return (
    <div className="bg-slate-900 rounded-xl border border-slate-800 p-5 flex flex-col h-full space-y-4">
      <h3 className="text-sm font-semibold text-slate-200 uppercase tracking-wider flex-shrink-0">
        Model
      </h3>

      <p className="text-sm text-slate-400 leading-relaxed flex-shrink-0">{info?.description}</p>

      {/* Spec grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 flex-shrink-0">
        {specs.map(s => (
          <div key={s.label} className="bg-slate-800 rounded-lg px-3 py-2">
            <div className="text-xs text-slate-500">{s.label}</div>
            <div className="text-sm text-white font-medium mt-0.5">{s.value}</div>
          </div>
        ))}
      </div>

      {/* Architecture */}
      {info?.architecture && (
        <div className="bg-slate-800/50 rounded-lg px-3 py-2 flex-shrink-0">
          <span className="text-xs text-slate-500">Architecture: </span>
          <span className="text-xs text-slate-300">{info.architecture}</span>
        </div>
      )}

      {/* Strengths — flex-1 pushes it down and fills remaining space */}
      {info?.strengths && (
        <div className="space-y-1.5 flex-1">
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
