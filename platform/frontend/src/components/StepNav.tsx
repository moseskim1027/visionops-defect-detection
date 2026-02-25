import type { Step } from '../types'

interface Props {
  currentStep: Step
  dataPrepComplete: boolean
  dataCardComplete: boolean
  trainingComplete: boolean
  onStepChange: (step: Step) => void
}

const steps: { id: Step; label: string; description: string }[] = [
  { id: 'data',     label: 'Prepare Data', description: 'Convert & validate dataset' },
  { id: 'datacard', label: 'Data Card',    description: 'Class distribution & samples' },
  { id: 'training', label: 'Training',     description: 'Configure & launch model training' },
  { id: 'model',    label: 'Evaluation',   description: 'Inspect metrics & training history' },
  { id: 'analysis', label: 'Analysis',     description: 'Per-class AP50 & challenging cases' },
]

export default function StepNav({
  currentStep, dataPrepComplete, dataCardComplete, trainingComplete, onStepChange,
}: Props) {
  const isUnlocked = (id: Step) => {
    if (id === 'data')     return true
    if (id === 'datacard') return dataPrepComplete
    if (id === 'training') return dataCardComplete
    if (id === 'model')    return trainingComplete
    if (id === 'analysis') return trainingComplete
    return false
  }

  const isComplete = (id: Step) => {
    if (id === 'data')     return dataPrepComplete
    if (id === 'datacard') return dataCardComplete
    if (id === 'training') return trainingComplete
    return false
  }

  return (
    <div className="bg-slate-900 border-b border-slate-800">
      <div className="max-w-6xl mx-auto px-6">
        <nav className="flex">
          {steps.map((step, idx) => {
            const active    = currentStep === step.id
            const complete  = isComplete(step.id)
            const unlocked  = isUnlocked(step.id)

            return (
              <button
                key={step.id}
                onClick={() => unlocked && onStepChange(step.id)}
                disabled={!unlocked}
                className={`
                  relative flex items-center gap-3 px-6 py-4 text-left transition-colors
                  ${active ? 'text-white' : unlocked ? 'text-slate-400 hover:text-slate-200' : 'text-slate-600 cursor-not-allowed'}
                `}
              >
                <div className={`
                  w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0
                  ${active ? 'bg-indigo-600 text-white' : complete ? 'bg-green-600 text-white' : unlocked ? 'bg-slate-700 text-slate-300' : 'bg-slate-800 text-slate-600'}
                `}>
                  {complete ? (
                    <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                    </svg>
                  ) : (
                    idx + 1
                  )}
                </div>

                <div>
                  <div className="text-sm font-medium leading-none">{step.label}</div>
                  <div className="text-xs text-slate-500 mt-0.5">{step.description}</div>
                </div>

                {active && (
                  <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-indigo-500 rounded-t" />
                )}

                {idx < steps.length - 1 && (
                  <div className="absolute right-0 top-1/2 -translate-y-1/2 translate-x-2 text-slate-700 pointer-events-none z-10">
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                )}
              </button>
            )
          })}
        </nav>
      </div>
    </div>
  )
}
