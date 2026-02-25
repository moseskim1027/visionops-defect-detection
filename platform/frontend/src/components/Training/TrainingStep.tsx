import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../../api/client'
import type { EpochResult, ModelInfo, TrainingConfig, TrainingStatus } from '../../types'
import ModelOverview from './ModelOverview'
import TrainingConfigPanel from './TrainingConfig'
import TrainingProgress from './TrainingProgress'

const POLL_INTERVAL = 3000

interface Props {
  onComplete: () => void
}

export default function TrainingStep({ onComplete }: Props) {
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null)
  const [config, setConfig] = useState<TrainingConfig | null>(null)
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus | null>(null)
  const [epochResults, setEpochResults] = useState<EpochResult[]>([])
  const [infoLoading, setInfoLoading] = useState(true)
  const [showConfig, setShowConfig] = useState(false)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    loadInitialData()
    checkExistingTraining()
  }, [])

  const loadInitialData = async () => {
    setInfoLoading(true)
    try {
      const [info, cfg] = await Promise.all([
        api.getModelInfo(),
        api.getTrainingConfig(),
      ])
      setModelInfo(info)
      setConfig(cfg)
    } finally {
      setInfoLoading(false)
    }
  }

  const checkExistingTraining = async () => {
    try {
      const status: TrainingStatus = await api.getTrainingStatus()
      setTrainingStatus(status)
      if (status.status === 'running') startPolling()
      if (status.status === 'completed' || status.status === 'failed') {
        const res = await api.getEpochResults()
        setEpochResults(res.results ?? [])
      }
    } catch {/* ignore */}
  }

  const startPolling = useCallback(() => {
    if (pollRef.current) return
    pollRef.current = setInterval(async () => {
      try {
        const [status, epochRes] = await Promise.all([
          api.getTrainingStatus() as Promise<TrainingStatus>,
          api.getEpochResults(),
        ])
        setTrainingStatus(status)
        setEpochResults(epochRes.results ?? [])

        if (status.status !== 'running') {
          clearInterval(pollRef.current!)
          pollRef.current = null
          const final = await api.getEpochResults()
          setEpochResults(final.results ?? [])
        }
      } catch {/* ignore */}
    }, POLL_INTERVAL)
  }, [])

  useEffect(() => {
    return () => { if (pollRef.current) clearInterval(pollRef.current) }
  }, [])

  const handleStart = async (cfg: TrainingConfig) => {
    try {
      setEpochResults([])
      setShowConfig(false)
      const { products, ...configFields } = cfg
      await api.startTraining({
        config: configFields,
        products: products && products.length > 0 ? products : undefined,
      })
      const status: TrainingStatus = await api.getTrainingStatus()
      setTrainingStatus(status)
      startPolling()
    } catch (e: any) {
      console.error('Failed to start training', e)
    }
  }

  const handleStop = async () => {
    try {
      await api.stopTraining()
      const status: TrainingStatus = await api.getTrainingStatus()
      setTrainingStatus(status)
      if (pollRef.current) {
        clearInterval(pollRef.current)
        pollRef.current = null
      }
    } catch (e: any) {
      console.error('Failed to stop training', e)
    }
  }

  const isTraining = trainingStatus?.status === 'running'
  const showProgress = !showConfig && (
    isTraining
    || trainingStatus?.status === 'completed'
    || trainingStatus?.status === 'failed'
  )

  return (
    <div className="max-w-6xl mx-auto px-6 py-8 space-y-8">
      <div>
        <h2 className="text-2xl font-bold text-white">Training</h2>
        <p className="text-slate-400 mt-1 text-sm">
          Configure and launch YOLOv8n training, then monitor progress in real time.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <ModelOverview info={modelInfo} loading={infoLoading} />

        {showProgress ? (
          <TrainingProgress
            status={trainingStatus}
            epochResults={epochResults}
            onStop={handleStop}
            onComplete={onComplete}
            onReconfigure={() => setShowConfig(true)}
          />
        ) : (
          <TrainingConfigPanel
            config={config}
            disabled={isTraining}
            onStart={handleStart}
          />
        )}
      </div>
    </div>
  )
}
