import { useEffect, useState } from 'react'
import { api } from '../../api/client'
import type { ClassMetric, PoorSample } from '../../types'
import ClassMetrics from '../ModelCard/ClassMetrics'
import PoorSamples from '../ModelCard/PoorSamples'

export default function AnalysisStep() {
  const [classMetrics, setClassMetrics] = useState<ClassMetric[]>([])
  const [poorSamples, setPoorSamples] = useState<PoorSample[]>([])
  const [metricsLoading, setMetricsLoading] = useState(true)
  const [samplesLoading, setSamplesLoading] = useState(true)
  const [metricsError, setMetricsError] = useState<string | null>(null)
  const [selectedClass, setSelectedClass] = useState<string | null>(null)

  useEffect(() => {
    loadClassMetrics()
  }, [])

  useEffect(() => {
    loadPoorSamples(selectedClass)
  }, [selectedClass])

  const loadClassMetrics = async () => {
    setMetricsLoading(true)
    setMetricsError(null)
    try {
      const res = await api.getClassMetrics()
      if (res.error) {
        setMetricsError(res.error)
      } else {
        setClassMetrics(res.class_metrics ?? [])
      }
    } catch (e: any) {
      setMetricsError(e.message)
    } finally {
      setMetricsLoading(false)
    }
  }

  const loadPoorSamples = async (className?: string | null) => {
    setSamplesLoading(true)
    try {
      const res = await api.getPoorSamples(className ?? undefined)
      setPoorSamples(res.samples ?? [])
    } finally {
      setSamplesLoading(false)
    }
  }

  return (
    <div className="max-w-6xl mx-auto px-6 py-8 space-y-8">
      {/* Title */}
      <div className="flex items-start justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white">Analysis</h2>
          <p className="text-slate-400 mt-1 text-sm">
            Per-class AP50 and challenging cases — click a class to filter samples.
          </p>
        </div>
        <button
          onClick={() => { loadClassMetrics(); loadPoorSamples(selectedClass) }}
          className="px-3 py-1.5 text-xs text-slate-400 hover:text-white border border-slate-700 hover:border-slate-500 rounded-md transition-colors"
        >
          Refresh
        </button>
      </div>

      {metricsError && (
        <div className="bg-amber-950/40 border border-amber-800/50 rounded-lg p-4 text-sm text-amber-300">
          <span className="font-medium">Note: </span>
          Per-class metrics require loading model weights.{' '}
          {metricsError.includes('No trained model') ? (
            'No trained model weights found. Complete a training run first.'
          ) : (
            <code className="text-xs">{metricsError}</code>
          )}
        </div>
      )}

      {/* Side-by-side: class AP50 left, challenging cases right */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left: Per-class AP50 */}
        <ClassMetrics
          classMetrics={classMetrics}
          loading={metricsLoading}
          selectedClass={selectedClass}
          onClassClick={setSelectedClass}
        />

        {/* Right: Challenging Cases */}
        <div className="bg-slate-900 rounded-xl border border-slate-800 p-5 space-y-3">
          <div className="flex items-start justify-between gap-3">
            <div>
              <h3 className="text-sm font-semibold text-slate-200 uppercase tracking-wider">
                Challenging Cases
              </h3>
              {selectedClass ? (
                <div className="flex items-center gap-2 mt-1.5">
                  <span className="text-xs bg-indigo-900/50 text-indigo-300 border border-indigo-700/50 px-2 py-0.5 rounded-full">
                    {selectedClass}
                  </span>
                  <button
                    onClick={() => setSelectedClass(null)}
                    className="text-xs text-slate-500 hover:text-slate-300 transition-colors"
                  >
                    Clear
                  </button>
                </div>
              ) : (
                <p className="text-xs text-slate-500 mt-0.5">Click image to expand</p>
              )}
            </div>
            <div className="flex items-center gap-3 text-xs flex-shrink-0">
              <span className="flex items-center gap-1.5">
                <span className="w-3 h-1.5 rounded-sm bg-green-400 inline-block" />
                <span className="text-slate-400">Ground truth</span>
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-3 h-1.5 rounded-sm bg-orange-400 inline-block" />
                <span className="text-slate-400">Predicted</span>
              </span>
            </div>
          </div>
          <PoorSamples samples={poorSamples} loading={samplesLoading} />
        </div>
      </div>
    </div>
  )
}
