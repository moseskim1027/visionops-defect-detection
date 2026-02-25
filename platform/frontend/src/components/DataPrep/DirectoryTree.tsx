interface Props {
  categories: string[]
  rawDir: string
}

export default function DirectoryTree({ categories, rawDir }: Props) {
  const display = categories.length > 0 ? categories : [
    'Cable', 'Capacitor', 'Casting', 'Console',
    'Cylinder', 'Electronics', 'Groove', 'Hemisphere',
    'Lens', 'PCB_1', 'PCB_2', 'Ring', 'Screw', 'Wood',
  ]

  return (
    <div className="font-mono text-sm bg-slate-950 rounded-lg border border-slate-800 p-4 overflow-auto max-h-72">
      <div className="text-slate-300">
        <TreeLine text="data/" color="text-indigo-400 font-semibold" />
        <TreeLine text="└── raw/" indent={0} />
        <TreeLine text="    └── vision/" indent={0} badge={rawDir} color="text-amber-400" />
        {display.map((cat, i) => {
          const isLast = i === display.length - 1
          const prefix = isLast ? '        └── ' : '        ├── '
          const childPrefix = isLast ? '             ' : '        │   '
          return (
            <div key={cat}>
              <TreeLine text={`${prefix}${cat}/`} color="text-sky-400" />
              <TreeLine text={`${childPrefix}├── train/`} />
              <TreeLine text={`${childPrefix}├── val/`} />
              <TreeLine text={`${childPrefix}└── _annotations.coco.json`} color="text-slate-500" />
            </div>
          )
        })}
      </div>
    </div>
  )
}

function TreeLine({
  text,
  indent = 0,
  color = 'text-slate-400',
  badge,
}: {
  text: string
  indent?: number
  color?: string
  badge?: string
}) {
  return (
    <div className="flex items-center gap-2 leading-6" style={{ paddingLeft: indent * 12 }}>
      <span className={color}>{text}</span>
      {badge && (
        <span className="text-xs bg-amber-900/40 text-amber-400 border border-amber-800/50 px-1.5 py-0.5 rounded">
          {badge}
        </span>
      )}
    </div>
  )
}
