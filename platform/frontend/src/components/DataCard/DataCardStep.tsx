import { useEffect, useState } from 'react'
import { api } from '../../api/client'
import type { AnnotatedSample, ClassCount } from '../../types'
import AnnotatedImages from '../DataPrep/AnnotatedImages'
import ClassDistribution from './ClassDistribution'

interface Props {
  onComplete: () => void
}

const N_SAMPLES = 9

export default function DataCardStep({ onComplete }: Props) {
  const [distribution, setDistribution] = useState<ClassCount[]>([])
  const [samples, setSamples] = useState<AnnotatedSample[]>([])
  const [distLoading, setDistLoading] = useState(true)
  const [samplesLoading, setSamplesLoading] = useState(true)
  const [selectedClass, setSelectedClass] = useState<string | null>(null)

  useEffect(() => {
    loadDistribution()
  }, [])

  useEffect(() => {
    loadSamples(selectedClass)
  }, [selectedClass])

  const loadDistribution = async () => {
    setDistLoading(true)
    try {
      const card = await api.getDataCard()
      setDistribution(card.distribution ?? [])
    } catch {/* ignore */} finally {
      setDistLoading(false)
    }
  }

  const loadSamples = async (className: string | null) => {
    setSamplesLoading(true)
    try {
      const data = await api.getAnnotatedSamples(N_SAMPLES, className ?? undefined)
      setSamples(data.samples ?? [])
    } catch {/* ignore */} finally {
      setSamplesLoading(false)
    }
  }

  return (
    <div className="max-w-6xl mx-auto px-6 py-8 space-y-8">
      <div>
        <h2 className="text-2xl font-bold text-white">Data Card</h2>
        <p className="text-slate-400 mt-1 text-sm">
          Explore class distribution and ground truth annotations in the prepared dataset.
        </p>
      </div>

      {/* Side-by-side: distribution left, annotations right — equal height via CSS grid stretch */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left */}
        <ClassDistribution
          distribution={distribution}
          loading={distLoading}
          selectedClass={selectedClass}
          onClassClick={setSelectedClass}
        />

        {/* Right */}
        <div className="bg-slate-900 rounded-xl border border-slate-800 p-5 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-slate-200 uppercase tracking-wider">
              Ground Truth Annotations
            </h3>
            <div className="flex items-center gap-2">
              {selectedClass && (
                <span className="text-xs bg-indigo-900/50 text-indigo-300 border border-indigo-700/50 px-2 py-0.5 rounded-full">
                  {selectedClass}
                </span>
              )}
              <span className="text-xs text-slate-500">Click image to expand</span>
            </div>
          </div>
          <AnnotatedImages samples={samples} loading={samplesLoading} cols={3} />
        </div>
      </div>

      {/* Continue */}
      <div className="flex justify-end">
        <button
          onClick={onComplete}
          className="px-6 py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-medium rounded-lg transition-colors flex items-center gap-2"
        >
          Continue to Training
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
        </button>
      </div>
    </div>
  )
}
