import { useEffect, useState } from 'react'
import { api } from '../../api/client'
import type { ClassMetric, ModelCard, PoorSample } from '../../types'
import MetricsSummary from './MetricsSummary'
import ClassMetrics from './ClassMetrics'
import PoorSamples from './PoorSamples'

export default function ModelCardStep() {
  const [card, setCard] = useState<ModelCard | null>(null)
  const [classMetrics, setClassMetrics] = useState<ClassMetric[]>([])
  const [poorSamples, setPoorSamples] = useState<PoorSample[]>([])
  const [cardLoading, setCardLoading] = useState(true)
  const [metricsLoading, setMetricsLoading] = useState(true)
  const [samplesLoading, setSamplesLoading] = useState(true)
  const [metricsError, setMetricsError] = useState<string | null>(null)

  useEffect(() => {
    loadCard()
    loadClassMetrics()
    loadPoorSamples()
  }, [])

  const loadCard = async () => {
    setCardLoading(true)
    try {
      const c = await api.getModelCard()
      setCard(c.error ? null : c)
    } finally {
      setCardLoading(false)
    }
  }

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

  const loadPoorSamples = async () => {
    setSamplesLoading(true)
    try {
      const res = await api.getPoorSamples()
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
          <h2 className="text-2xl font-bold text-white">Evaluation</h2>
          <p className="text-slate-400 mt-1 text-sm">
            Post-training evaluation — overall metrics, per-class AP50, and failure cases.
          </p>
        </div>
        <button
          onClick={() => { loadCard(); loadClassMetrics(); loadPoorSamples() }}
          className="px-3 py-1.5 text-xs text-slate-400 hover:text-white border border-slate-700 hover:border-slate-500 rounded-md transition-colors"
        >
          Refresh
        </button>
      </div>

      {/* Overall metrics */}
      <section className="space-y-3">
        <SectionTitle>Overall Performance</SectionTitle>
        <MetricsSummary card={card} loading={cardLoading} />
      </section>

      {/* Error state for class metrics */}
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

      {/* Per-class AP50 */}
      <section className="space-y-3">
        <SectionTitle>Class-level Analysis</SectionTitle>
        <ClassMetrics classMetrics={classMetrics} loading={metricsLoading} />
      </section>

      {/* Poor samples */}
      <section className="space-y-3">
        <div className="flex items-center justify-between">
          <SectionTitle>Challenging Cases</SectionTitle>
          <div className="flex items-center gap-3 text-xs">
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
      </section>
    </div>
  )
}

function SectionTitle({ children }: { children: React.ReactNode }) {
  return <h3 className="text-lg font-semibold text-white">{children}</h3>
}
