import { useState } from 'react'
import type { TrainingConfig } from '../../types'
import { api } from '../../api/client'

interface Props {
  config: TrainingConfig | null
  disabled: boolean
  onStart: (config: TrainingConfig) => void
}

export default function TrainingConfigPanel({ config, disabled, onStart }: Props) {
  const [form, setForm] = useState<Partial<TrainingConfig>>({})
  const [saving, setSaving] = useState(false)

  const effective = { ...config, ...form } as TrainingConfig

  const set = (key: keyof TrainingConfig, value: string | number) => {
    setForm(f => ({ ...f, [key]: value }))
  }

  const handleStart = async () => {
    setSaving(true)
    try {
      if (Object.keys(form).length > 0) {
        await api.updateTrainingConfig(form)
      }
      onStart(effective)
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="bg-slate-900 rounded-xl border border-slate-800 p-5 space-y-5">
      <h3 className="text-sm font-semibold text-slate-200 uppercase tracking-wider">
        Training Configuration
      </h3>

      <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
        <NumberField
          label="Epochs"
          value={effective.epochs ?? 10}
          min={1}
          max={300}
          onChange={v => set('epochs', v)}
          disabled={disabled}
        />
        <NumberField
          label="Batch size"
          value={effective.batch ?? 8}
          min={1}
          max={128}
          onChange={v => set('batch', v)}
          disabled={disabled}
        />
        <NumberField
          label="Learning rate"
          value={effective.lr0 ?? 0.01}
          step={0.001}
          min={0.0001}
          max={0.1}
          onChange={v => set('lr0', v)}
          disabled={disabled}
          isFloat
        />
        <NumberField
          label="Patience"
          value={effective.patience ?? 5}
          min={1}
          max={100}
          onChange={v => set('patience', v)}
          disabled={disabled}
        />
        <NumberField
          label="Image size"
          value={effective.imgsz ?? 640}
          min={320}
          max={1280}
          step={32}
          onChange={v => set('imgsz', v)}
          disabled={disabled}
        />
        <div>
          <label className="block text-xs text-slate-400 mb-1.5">Device</label>
          <select
            value={effective.device ?? 'cpu'}
            onChange={e => set('device', e.target.value)}
            disabled={disabled}
            className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-indigo-500 disabled:opacity-50"
          >
            <option value="cpu">CPU</option>
            <option value="cuda:0">CUDA GPU 0</option>
            <option value="mps">Apple MPS</option>
          </select>
        </div>
      </div>

      <button
        onClick={handleStart}
        disabled={disabled || saving}
        className="w-full py-3 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed text-white font-semibold rounded-lg transition-colors flex items-center justify-center gap-2"
      >
        {saving ? (
          <><Spinner /> Saving config...</>
        ) : disabled ? (
          <>
            <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
            Training in progress...
          </>
        ) : (
          <>
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            Start Training
          </>
        )}
      </button>
    </div>
  )
}

function NumberField({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
  disabled,
  isFloat = false,
}: {
  label: string
  value: number
  min: number
  max: number
  step?: number
  onChange: (v: number) => void
  disabled: boolean
  isFloat?: boolean
}) {
  return (
    <div>
      <label className="block text-xs text-slate-400 mb-1.5">{label}</label>
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={e => onChange(isFloat ? parseFloat(e.target.value) : parseInt(e.target.value))}
        disabled={disabled}
        className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-indigo-500 disabled:opacity-50"
      />
    </div>
  )
}

function Spinner() {
  return (
    <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
    </svg>
  )
}
