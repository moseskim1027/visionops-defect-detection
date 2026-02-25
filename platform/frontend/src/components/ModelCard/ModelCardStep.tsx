import { useEffect, useState } from 'react'
import { api } from '../../api/client'
import type { EpochResult, ModelCard } from '../../types'
import MetricsSummary from './MetricsSummary'
import TrainingCharts from './TrainingCharts'

interface Props {
  onComplete: () => void
}

export default function ModelCardStep({ onComplete }: Props) {
  const [card, setCard] = useState<ModelCard | null>(null)
  const [epochResults, setEpochResults] = useState<EpochResult[]>([])
  const [cardLoading, setCardLoading] = useState(true)
  const [epochsLoading, setEpochsLoading] = useState(true)

  useEffect(() => {
    loadCard()
    loadEpochResults()
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

  const loadEpochResults = async () => {
    setEpochsLoading(true)
    try {
      const res = await api.getEpochResults()
      setEpochResults(res.results ?? [])
    } finally {
      setEpochsLoading(false)
    }
  }

  return (
    <div className="max-w-6xl mx-auto px-6 py-8 space-y-8">
      {/* Title */}
      <div className="flex items-start justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white">Evaluation</h2>
          <p className="text-slate-400 mt-1 text-sm">
            Post-training evaluation — overall metrics and training history.
          </p>
        </div>
        <button
          onClick={() => { loadCard(); loadEpochResults() }}
          className="px-3 py-1.5 text-xs text-slate-400 hover:text-white border border-slate-700 hover:border-slate-500 rounded-md transition-colors"
        >
          Refresh
        </button>
      </div>

      {/* Training history charts */}
      <section className="space-y-3">
        <SectionTitle>Training History</SectionTitle>
        <TrainingCharts epochResults={epochResults} loading={epochsLoading} />
      </section>

      {/* Overall metrics */}
      <section className="space-y-3">
        <SectionTitle>Overall Performance</SectionTitle>
        <MetricsSummary card={card} loading={cardLoading} />
      </section>

      {/* Continue */}
      <div className="flex justify-end">
        <button
          onClick={onComplete}
          className="px-6 py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-medium rounded-lg transition-colors flex items-center gap-2"
        >
          View Analysis
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
        </button>
      </div>
    </div>
  )
}

function SectionTitle({ children }: { children: React.ReactNode }) {
  return <h3 className="text-lg font-semibold text-white">{children}</h3>
}
