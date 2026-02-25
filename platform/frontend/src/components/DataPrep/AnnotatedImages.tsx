import { useEffect, useState } from 'react'
import type { AnnotatedSample } from '../../types'

interface Props {
  samples: AnnotatedSample[]
  loading: boolean
  cols?: 2 | 3 | 4
}

const GRID_COLS = {
  2: 'grid-cols-2',
  3: 'grid-cols-3',
  4: '{GRID_COLS[cols]}',
} as const

export default function AnnotatedImages({ samples, loading, cols = 4 }: Props) {
  const [lightboxIdx, setLightboxIdx] = useState<number | null>(null)

  // Close on Escape, navigate with arrow keys
  useEffect(() => {
    if (lightboxIdx === null) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setLightboxIdx(null)
      if (e.key === 'ArrowRight') setLightboxIdx(i => (i !== null ? Math.min(i + 1, samples.length - 1) : null))
      if (e.key === 'ArrowLeft') setLightboxIdx(i => (i !== null ? Math.max(i - 1, 0) : null))
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [lightboxIdx, samples.length])

  if (loading) {
    return (
      <div className={`grid ${GRID_COLS[cols]} gap-3`}>
        {Array.from({ length: cols * 3 }).map((_, i) => (
          <div key={i} className="aspect-square bg-slate-800 rounded-lg animate-pulse" />
        ))}
      </div>
    )
  }

  if (!samples.length) {
    return (
      <div className="text-slate-500 text-sm text-center py-12">
        No samples available. Prepare the dataset first.
      </div>
    )
  }

  const active = lightboxIdx !== null ? samples[lightboxIdx] : null

  return (
    <>
      <div className={`grid ${GRID_COLS[cols]} gap-3`}>
        {samples.map((sample, i) => (
          <button
            key={i}
            onClick={() => setLightboxIdx(i)}
            className="group relative bg-slate-800 rounded-lg overflow-hidden border border-slate-700 hover:border-indigo-500 transition-colors text-left focus:outline-none focus:ring-2 focus:ring-indigo-500"
          >
            <img
              src={`data:image/jpeg;base64,${sample.image_b64}`}
              alt={sample.filename}
              className="w-full aspect-square object-cover"
              loading="lazy"
            />
            {/* Expand hint */}
            <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
            <div className="absolute bottom-0 left-0 right-0 p-2 opacity-0 group-hover:opacity-100 transition-opacity">
              <div className="text-white text-xs font-medium truncate">{sample.filename}</div>
              <div className="text-slate-300 text-xs">{sample.num_annotations} annotations</div>
            </div>
            {/* Expand icon */}
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 transition-opacity">
              <div className="bg-black/60 rounded-full p-1.5">
                <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                    d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
                </svg>
              </div>
            </div>
            {/* Category badge */}
            <div className="absolute top-2 left-2">
              <span className="bg-black/60 text-xs text-white px-1.5 py-0.5 rounded">
                {sample.category}
              </span>
            </div>
            {sample.num_annotations > 0 && (
              <div className="absolute top-2 right-2">
                <span className="bg-indigo-600/80 text-xs text-white px-1.5 py-0.5 rounded">
                  {sample.num_annotations}
                </span>
              </div>
            )}
          </button>
        ))}
      </div>

      {/* Lightbox */}
      {active && lightboxIdx !== null && (
        <div
          className="fixed inset-0 z-50 bg-black/90 flex items-center justify-center"
          onClick={() => setLightboxIdx(null)}
        >
          {/* Close */}
          <button
            onClick={() => setLightboxIdx(null)}
            className="absolute top-4 right-4 text-slate-400 hover:text-white transition-colors"
          >
            <svg className="w-7 h-7" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>

          {/* Prev */}
          {lightboxIdx > 0 && (
            <button
              onClick={e => { e.stopPropagation(); setLightboxIdx(lightboxIdx - 1) }}
              className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400 hover:text-white transition-colors bg-black/40 rounded-full p-2"
            >
              <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
              </svg>
            </button>
          )}

          {/* Image */}
          <div
            className="max-w-3xl max-h-[80vh] flex flex-col items-center gap-3"
            onClick={e => e.stopPropagation()}
          >
            <img
              src={`data:image/jpeg;base64,${active.image_b64}`}
              alt={active.filename}
              className="max-w-full max-h-[70vh] object-contain rounded-lg border border-slate-700"
            />
            <div className="flex items-center gap-4 text-sm">
              <span className="text-slate-300 font-medium">{active.filename}</span>
              <span className="bg-black/60 text-xs text-white px-2 py-0.5 rounded">{active.category}</span>
              <span className="text-slate-400">{active.num_annotations} annotations</span>
              <span className="text-slate-600">{lightboxIdx + 1} / {samples.length}</span>
            </div>
          </div>

          {/* Next */}
          {lightboxIdx < samples.length - 1 && (
            <button
              onClick={e => { e.stopPropagation(); setLightboxIdx(lightboxIdx + 1) }}
              className="absolute right-4 top-1/2 -translate-y-1/2 text-slate-400 hover:text-white transition-colors bg-black/40 rounded-full p-2"
            >
              <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            </button>
          )}
        </div>
      )}
    </>
  )
}
