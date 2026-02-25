import { useState } from 'react'
import type { Step } from './types'
import StepNav from './components/StepNav'
import DataPrepStep from './components/DataPrep/DataPrepStep'
import DataCardStep from './components/DataCard/DataCardStep'
import TrainingStep from './components/Training/TrainingStep'
import ModelCardStep from './components/ModelCard/ModelCardStep'
import AnalysisStep from './components/Analysis/AnalysisStep'
import DeployStep from './components/Deploy/DeployStep'

export default function App() {
  const [currentStep, setCurrentStep] = useState<Step>('data')
  const [dataPrepComplete, setDataPrepComplete] = useState(false)
  const [dataCardComplete, setDataCardComplete] = useState(false)
  const [trainingComplete, setTrainingComplete] = useState(false)
  const [evaluationComplete, setEvaluationComplete] = useState(false)
  const [analysisComplete, setAnalysisComplete] = useState(false)

  return (
    <div className="flex flex-col min-h-screen bg-slate-950">
      {/* Header */}
      <header className="bg-slate-900 border-b border-slate-800 px-6 py-4 flex items-center gap-4">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-indigo-600 rounded-lg flex items-center justify-center">
            <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
            </svg>
          </div>
          <div>
            <h1 className="text-white font-semibold text-base leading-none">VisionOps</h1>
            <p className="text-slate-500 text-xs mt-0.5">Defect Detection Platform</p>
          </div>
        </div>

        <div className="flex-1" />

        <div className="flex items-center gap-2">
          {[
            { label: 'MLflow', href: 'http://localhost:5001' },
            { label: 'Grafana', href: 'http://localhost:3000' },
            { label: 'Prometheus', href: 'http://localhost:9090' },
          ].map(({ label, href }) => (
            <a
              key={label}
              href={href}
              target="_blank"
              rel="noreferrer"
              className="px-3 py-1.5 text-xs text-slate-400 hover:text-white border border-slate-700 hover:border-slate-500 rounded-md transition-colors"
            >
              {label} ↗
            </a>
          ))}
        </div>
      </header>

      {/* Step navigation */}
      <StepNav
        currentStep={currentStep}
        dataPrepComplete={dataPrepComplete}
        dataCardComplete={dataCardComplete}
        trainingComplete={trainingComplete}
        evaluationComplete={evaluationComplete}
        analysisComplete={analysisComplete}
        onStepChange={setCurrentStep}
      />

      {/* Main content */}
      <main className="flex-1 overflow-auto">
        {currentStep === 'data' && (
          <DataPrepStep
            onComplete={() => {
              setDataPrepComplete(true)
              setCurrentStep('datacard')
            }}
          />
        )}
        {currentStep === 'datacard' && (
          <DataCardStep
            onComplete={() => {
              setDataCardComplete(true)
              setCurrentStep('training')
            }}
          />
        )}
        {currentStep === 'training' && (
          <TrainingStep
            onComplete={() => {
              setTrainingComplete(true)
              setCurrentStep('model')
            }}
          />
        )}
        {currentStep === 'model' && (
          <ModelCardStep onComplete={() => { setEvaluationComplete(true); setCurrentStep('analysis') }} />
        )}
        {currentStep === 'analysis' && (
          <AnalysisStep onComplete={() => { setAnalysisComplete(true); setCurrentStep('deploy') }} />
        )}
        {currentStep === 'deploy' && <DeployStep />}
      </main>
    </div>
  )
}
