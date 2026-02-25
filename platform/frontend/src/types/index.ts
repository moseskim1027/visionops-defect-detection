export type Step = 'data' | 'training' | 'model'

export interface DirectoryInfo {
  raw_dir: string
  processed_dir: string
  categories: string[]
  is_prepared: boolean
  num_train_images: number
  num_val_images: number
  prep_state: PrepState
}

export interface PrepState {
  status: 'idle' | 'running' | 'completed' | 'failed'
  started_at: number | null
  completed_at: number | null
  message: string
  error: string | null
}

export interface AnnotatedSample {
  filename: string
  category: string
  image_b64: string
  annotations: { class_id: number; class_name: string }[]
  num_annotations: number
}

export interface ModelInfo {
  name: string
  full_name: string
  parameters: string
  gflops: string
  input_size: string
  task: string
  framework: string
  architecture: string
  description: string
  strengths: string[]
  training_config: TrainingConfig
}

export interface TrainingConfig {
  epochs: number
  batch: number
  lr0: number
  patience: number
  workers: number
  device: string
  imgsz: number
  experiment_name: string
}

export interface ClassCount {
  class_id: number
  class_name: string
  count: number
}

export interface TrainingStatus {
  status: 'idle' | 'running' | 'completed' | 'failed'
  run_id: string | null
  started_at: number | null
  completed_at: number | null
  error: string | null
  pid: number | null
  elapsed_seconds?: number
}

export interface MetricPoint {
  step: number
  value: number
  timestamp: number
}

export interface MLflowMetrics {
  run_id: string
  status: string
  metrics: Record<string, MetricPoint[]>
  params: Record<string, string>
  start_time: number
  end_time: number | null
  error?: string
}

export interface MLflowRun {
  run_id: string
  status: string
  start_time: number
  end_time: number | null
  metrics: Record<string, number>
  params: Record<string, string>
}

export interface OverallMetrics {
  precision: number
  recall: number
  map50: number
  map50_95: number
}

export interface ClassMetric {
  class_id: number
  class_name: string
  ap50: number
  ap50_95: number
}

export interface ClassMetricsResponse {
  overall: OverallMetrics
  class_metrics: ClassMetric[]
  weights_path: string
  error?: string
}

export interface ModelCard {
  run_id: string
  status: string
  metrics: OverallMetrics
  params: Record<string, string>
  start_time: number
  end_time: number | null
  error?: string
}

export interface PoorSample {
  filename: string
  class_name: string
  ap50: number
  image_b64: string
}

export interface PredictionDistEntry {
  class_id: number
  class_name: string
  count: number
  percentage: number
}
