import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../../api/client'
import type {
  ClassCount, MLflowMetrics, ModelInfo, TrainingConfig, TrainingStatus,
} from '../../types'
import ModelOverview from './ModelOverview'
import DataCard from './DataCard'
import TrainingConfigPanel from './TrainingConfig'
import TrainingProgress from './TrainingProgress'

const GRAFANA_URL = import.meta.env.VITE_GRAFANA_URL ?? 'http://localhost:3000'
const MLFLOW_URL = import.meta.env.VITE_MLFLOW_URL ?? 'http://localhost:5001'
const POLL_INTERVAL = 3000

interface Props {
  onComplete: () => void
}

export default function TrainingStep({ onComplete }: Props) {
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null)
  const [config, setConfig] = useState<TrainingConfig | null>(null)
  const [distribution, setDistribution] = useState<ClassCount[]>([])
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus | null>(null)
  const [mlflowMetrics, setMlflowMetrics] = useState<MLflowMetrics | null>(null)
  const [infoLoading, setInfoLoading] = useState(true)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    loadInitialData()
    checkExistingTraining()
  }, [])

  const loadInitialData = async () => {
    setInfoLoading(true)
    try {
      const [info, cfg, card] = await Promise.all([
        api.getModelInfo(),
        api.getTrainingConfig(),
        api.getDataCard(),
      ])
      setModelInfo(info)
      setConfig(cfg)
      setDistribution(card.distribution ?? [])
    } finally {
      setInfoLoading(false)
    }
  }

  const checkExistingTraining = async () => {
    try {
      const status: TrainingStatus = await api.getTrainingStatus()
      setTrainingStatus(status)
      if (status.status === 'running') {
        startPolling()
      }
    } catch {/* ignore */}
  }

  const startPolling = useCallback(() => {
    if (pollRef.current) return
    pollRef.current = setInterval(async () => {
      try {
        const status: TrainingStatus = await api.getTrainingStatus()
        setTrainingStatus(status)

        if (status.run_id) {
          const m: MLflowMetrics = await api.getMLflowMetrics(status.run_id)
          if (!m.error) setMlflowMetrics(m)
        }

        if (status.status === 'completed') {
          clearInterval(pollRef.current!)
          pollRef.current = null
        } else if (status.status === 'failed') {
          clearInterval(pollRef.current!)
          pollRef.current = null
        }
      } catch {/* ignore */}
    }, POLL_INTERVAL)
  }, [])

  useEffect(() => {
    return () => { if (pollRef.current) clearInterval(pollRef.current) }
  }, [])

  const handleStart = async (cfg: TrainingConfig) => {
    try {
      await api.startTraining({ config: cfg })
      const status: TrainingStatus = await api.getTrainingStatus()
      setTrainingStatus(status)
      startPolling()
    } catch (e: any) {
      console.error('Failed to start training', e)
    }
  }

  const isTraining = trainingStatus?.status === 'running'
  const isComplete = trainingStatus?.status === 'completed'

  return (
    <div className="max-w-6xl mx-auto px-6 py-8 space-y-8">
      {/* Title */}
      <div>
        <h2 className="text-2xl font-bold text-white">Training</h2>
        <p className="text-slate-400 mt-1 text-sm">
          Configure and launch YOLOv8n training, then monitor progress in real time.
        </p>
      </div>

      {/* Row 1 — model overview + data card */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <ModelOverview info={modelInfo} loading={infoLoading} />
        <DataCard distribution={distribution} loading={infoLoading} />
      </div>

      {/* Row 2 — training config */}
      <TrainingConfigPanel
        config={config}
        disabled={isTraining}
        onStart={handleStart}
      />

      {/* Row 3 — progress (visible after training starts) */}
      {(isTraining || isComplete || trainingStatus?.status === 'failed') && (
        <div>
          <h3 className="text-lg font-semibold text-white mb-4">Training Progress</h3>
          <TrainingProgress
            status={trainingStatus}
            metrics={mlflowMetrics}
            grafanaUrl={GRAFANA_URL}
            mlflowUrl={MLFLOW_URL}
          />
        </div>
      )}

      {/* Continue button */}
      {isComplete && (
        <div className="flex justify-end">
          <button
            onClick={onComplete}
            className="px-6 py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-medium rounded-lg transition-colors flex items-center gap-2"
          >
            View Model Card
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
          </button>
        </div>
      )}
    </div>
  )
}
