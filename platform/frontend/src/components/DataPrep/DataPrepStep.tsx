import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../../api/client'
import type { AnnotatedSample, DirectoryInfo, PrepState } from '../../types'
import DirectoryTree from './DirectoryTree'
import AnnotatedImages from './AnnotatedImages'

interface Props {
  onComplete: () => void
}

export default function DataPrepStep({ onComplete }: Props) {
  const [dirInfo, setDirInfo] = useState<DirectoryInfo | null>(null)
  const [customDir, setCustomDir] = useState('')
  const [samples, setSamples] = useState<AnnotatedSample[]>([])
  const [samplesLoading, setSamplesLoading] = useState(false)
  const [preparing, setPreparing] = useState(false)
  const [prepState, setPrepState] = useState<PrepState | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const loadDirInfo = useCallback(async (dir?: string) => {
    try {
      const info = await api.getDirectoryInfo(dir)
      setDirInfo(info)
      setPrepState(info.prep_state)
      if (info.is_prepared && !samples.length) {
        loadSamples()
      }
    } catch {
      // API not yet reachable
    }
  }, [samples.length])

  const loadSamples = async () => {
    setSamplesLoading(true)
    try {
      const res = await api.getAnnotatedSamples(12)
      setSamples(res.samples ?? [])
    } finally {
      setSamplesLoading(false)
    }
  }

  useEffect(() => {
    loadDirInfo()
  }, [])

  // Poll during preparation
  useEffect(() => {
    if (preparing) {
      pollRef.current = setInterval(async () => {
        try {
          const state: PrepState = await api.getPreparationStatus()
          setPrepState(state)
          if (state.status === 'completed') {
            setPreparing(false)
            clearInterval(pollRef.current!)
            loadDirInfo(customDir || undefined)
            loadSamples()
          } else if (state.status === 'failed') {
            setPreparing(false)
            clearInterval(pollRef.current!)
          }
        } catch {/* ignore */}
      }, 1500)
    }
    return () => { if (pollRef.current) clearInterval(pollRef.current) }
  }, [preparing])

  const handlePrepare = async () => {
    setPreparing(true)
    setSamples([])
    try {
      await api.prepareDataset({ source_dir: customDir || undefined })
    } catch (e) {
      setPreparing(false)
    }
  }

  const isPrepared = dirInfo?.is_prepared || prepState?.status === 'completed'

  return (
    <div className="max-w-6xl mx-auto px-6 py-8 space-y-8">
      {/* Title */}
      <div>
        <h2 className="text-2xl font-bold text-white">Data Preparation</h2>
        <p className="text-slate-400 mt-1 text-sm">
          Convert COCO-format annotations from the VISION industrial dataset into YOLO format.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left — directory setup */}
        <div className="space-y-5">
          {/* Directory path */}
          <div className="bg-slate-900 rounded-xl border border-slate-800 p-5 space-y-4">
            <h3 className="text-sm font-semibold text-slate-200 uppercase tracking-wider">
              Data Source
            </h3>

            <div>
              <label className="block text-xs text-slate-400 mb-1.5">Raw data directory</label>
              <input
                type="text"
                value={customDir}
                onChange={e => setCustomDir(e.target.value)}
                onBlur={() => loadDirInfo(customDir || undefined)}
                placeholder={dirInfo?.raw_dir ?? 'data/raw/vision'}
                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
              />
              <p className="text-xs text-slate-500 mt-1">
                Default: <code className="text-slate-400">{dirInfo?.raw_dir ?? 'data/raw/vision'}</code>
              </p>
            </div>

            {/* Stats */}
            {dirInfo && (
              <div className="grid grid-cols-3 gap-3">
                <Stat label="Categories" value={dirInfo.categories.length || 14} />
                <Stat label="Train images" value={dirInfo.num_train_images} />
                <Stat label="Val images" value={dirInfo.num_val_images} />
              </div>
            )}

            {/* Status badge */}
            {isPrepared && (
              <div className="flex items-center gap-2 text-green-400 text-sm">
                <div className="w-2 h-2 bg-green-400 rounded-full" />
                Dataset prepared — {dirInfo?.num_train_images ?? '?'} train / {dirInfo?.num_val_images ?? '?'} val images
              </div>
            )}

            {/* Prepare button */}
            <button
              onClick={handlePrepare}
              disabled={preparing}
              className="w-full py-2.5 px-4 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed text-white text-sm font-medium rounded-lg transition-colors flex items-center justify-center gap-2"
            >
              {preparing ? (
                <>
                  <Spinner />
                  Preparing dataset...
                </>
              ) : isPrepared ? (
                'Re-prepare Dataset'
              ) : (
                'Prepare Dataset'
              )}
            </button>

            {/* Error */}
            {prepState?.status === 'failed' && prepState.error && (
              <div className="bg-red-950/40 border border-red-800/50 rounded-lg p-3 text-xs text-red-400 font-mono overflow-auto max-h-32">
                {prepState.error}
              </div>
            )}

            {/* Progress message */}
            {preparing && prepState?.message && (
              <div className="text-xs text-slate-400 flex items-center gap-2">
                <Spinner size="sm" />
                {prepState.message}
              </div>
            )}
          </div>

          {/* Directory tree */}
          <div className="bg-slate-900 rounded-xl border border-slate-800 p-5 space-y-3">
            <h3 className="text-sm font-semibold text-slate-200 uppercase tracking-wider">
              Expected Structure
            </h3>
            <DirectoryTree
              categories={dirInfo?.categories ?? []}
              rawDir={customDir || dirInfo?.raw_dir ?? 'data/raw/vision'}
            />
          </div>
        </div>

        {/* Right — annotated samples */}
        <div className="bg-slate-900 rounded-xl border border-slate-800 p-5 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-slate-200 uppercase tracking-wider">
              Ground Truth Annotations
            </h3>
            {isPrepared && (
              <button
                onClick={loadSamples}
                className="text-xs text-indigo-400 hover:text-indigo-300 transition-colors"
              >
                Reshuffle
              </button>
            )}
          </div>

          {!isPrepared && !samplesLoading ? (
            <div className="flex flex-col items-center justify-center py-16 text-slate-600 space-y-2">
              <svg className="w-12 h-12" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1}
                  d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
              <p className="text-sm">Sample images appear after preparation</p>
            </div>
          ) : (
            <AnnotatedImages samples={samples} loading={samplesLoading} />
          )}
        </div>
      </div>

      {/* Continue button */}
      {isPrepared && (
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
      )}
    </div>
  )
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div className="bg-slate-800 rounded-lg p-3 text-center">
      <div className="text-xl font-bold text-white">{value || '—'}</div>
      <div className="text-xs text-slate-400 mt-0.5">{label}</div>
    </div>
  )
}

function Spinner({ size = 'md' }: { size?: 'sm' | 'md' }) {
  const cls = size === 'sm' ? 'w-3 h-3' : 'w-4 h-4'
  return (
    <svg className={`${cls} animate-spin`} fill="none" viewBox="0 0 24 24">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
    </svg>
  )
}
