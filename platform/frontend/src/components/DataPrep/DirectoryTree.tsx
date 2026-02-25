import type { ProductDetail } from '../../types'

interface Props {
  productDetails: ProductDetail[]
  rawDir: string
}

export default function DirectoryTree({ productDetails, rawDir }: Props) {
  const compatCount = productDetails.filter(p => p.compatible).length
  const total = productDetails.length
  const allCompatible = total > 0 && compatCount === total

  return (
    <div className="space-y-4">
      {/* ── Actual structure ── */}
      <div>
        <div className="flex items-center justify-between mb-1.5">
          <span className="text-xs font-medium text-slate-300">Actual — <code className="text-slate-400">{rawDir}</code></span>
          {total > 0 && (
            <span className={`text-xs px-1.5 py-0.5 rounded ${
              allCompatible
                ? 'bg-green-900/40 text-green-400 border border-green-800/40'
                : 'bg-amber-900/40 text-amber-400 border border-amber-800/40'
            }`}>
              {compatCount}/{total} compatible
            </span>
          )}
        </div>

        <div className="font-mono text-xs bg-slate-950 rounded-lg border border-slate-800 p-3 overflow-auto max-h-60">
          {total === 0 ? (
            <span className="text-red-400">Directory not found or empty</span>
          ) : (
            <>
              <div className="text-slate-400">{rawDir.split('/').pop() ?? rawDir}/</div>
              {productDetails.map((p, i) => {
                const isLast = i === total - 1
                const prefix = isLast ? '└── ' : '├── '
                const issues: string[] = []
                if (!p.has_train) issues.push('missing train/')
                if (!p.has_val) issues.push('missing val/')
                if (!p.has_annotations) issues.push('missing annotations')
                return (
                  <div key={p.name} className="flex items-baseline gap-2 leading-5">
                    <span className={p.compatible ? 'text-sky-400' : 'text-amber-400'}>
                      {prefix}{p.name}/
                    </span>
                    {p.compatible ? (
                      <span className="text-green-500">✓</span>
                    ) : (
                      <span className="text-red-400 text-xs">{issues.join(', ')}</span>
                    )}
                  </div>
                )
              })}
            </>
          )}
        </div>
      </div>

      {/* ── Expected structure ── */}
      <div>
        <span className="text-xs font-medium text-slate-300 block mb-1.5">Expected Structure</span>
        <div className="font-mono text-xs bg-slate-950 rounded-lg border border-slate-800 p-3 overflow-auto text-slate-400">
          <div className="text-indigo-400">vision/</div>
          <div>├── <span className="text-sky-400">{'{Product}'}/</span></div>
          <div>│   ├── <span className="text-slate-300">train/</span></div>
          <div>│   │   ├── *.jpg</div>
          <div className="text-green-500/70">│   │   └── _annotations.coco.json</div>
          <div>│   └── <span className="text-slate-300">val/</span></div>
          <div>│       ├── *.jpg</div>
          <div className="text-green-500/70">│       └── _annotations.coco.json</div>
          <div>└── ...</div>
        </div>
      </div>
    </div>
  )
}
