import type { AnnotatedSample } from '../../types'

interface Props {
  samples: AnnotatedSample[]
  loading: boolean
}

export default function AnnotatedImages({ samples, loading }: Props) {
  if (loading) {
    return (
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
        {Array.from({ length: 12 }).map((_, i) => (
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

  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
      {samples.map((sample, i) => (
        <div
          key={i}
          className="group relative bg-slate-800 rounded-lg overflow-hidden border border-slate-700 hover:border-indigo-500 transition-colors"
        >
          <img
            src={`data:image/jpeg;base64,${sample.image_b64}`}
            alt={sample.filename}
            className="w-full aspect-square object-cover"
            loading="lazy"
          />
          {/* Overlay */}
          <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
          <div className="absolute bottom-0 left-0 right-0 p-2 opacity-0 group-hover:opacity-100 transition-opacity">
            <div className="text-white text-xs font-medium truncate">{sample.filename}</div>
            <div className="text-slate-300 text-xs">{sample.num_annotations} annotations</div>
          </div>
          {/* Category badge */}
          <div className="absolute top-2 left-2">
            <span className="bg-black/60 text-xs text-white px-1.5 py-0.5 rounded">
              {sample.category}
            </span>
          </div>
          {/* Annotation count badge */}
          {sample.num_annotations > 0 && (
            <div className="absolute top-2 right-2">
              <span className="bg-indigo-600/80 text-xs text-white px-1.5 py-0.5 rounded">
                {sample.num_annotations}
              </span>
            </div>
          )}
        </div>
      ))}
    </div>
  )
}
