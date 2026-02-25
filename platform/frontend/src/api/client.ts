const BASE = import.meta.env.VITE_API_URL ?? '/api'

async function get<T>(path: string, params?: Record<string, string | number | boolean>): Promise<T> {
  const url = new URL(`${BASE}${path}`, window.location.origin)
  if (params) {
    Object.entries(params).forEach(([k, v]) => url.searchParams.set(k, String(v)))
  }
  const res = await fetch(url.toString())
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return res.json()
}

async function post<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  })
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return res.json()
}

async function put<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return res.json()
}

export const api = {
  // ── Data ──────────────────────────────────────────────────────────────────
  getDirectoryInfo: (sourceDir?: string) =>
    get<any>('/data/directory-info', sourceDir ? { source_dir: sourceDir } : undefined),

  prepareDataset: (body: { source_dir?: string; processed_dir?: string }) =>
    post<any>('/data/prepare', body),

  getPreparationStatus: () => get<any>('/data/preparation-status'),

  getAnnotatedSamples: (nSamples = 12) =>
    get<any>('/data/samples', { n_samples: nSamples }),

  // ── Training ──────────────────────────────────────────────────────────────
  getModelInfo: () => get<any>('/training/model-info'),

  getDataCard: () => get<any>('/training/data-card'),

  getTrainingConfig: () => get<any>('/training/config'),

  updateTrainingConfig: (config: Partial<any>) => put<any>('/training/config', config),

  startTraining: (body?: { dataset_yaml?: string; config?: Partial<any> }) =>
    post<any>('/training/start', body ?? {}),

  getTrainingStatus: () => get<any>('/training/status'),

  stopTraining: () => post<any>('/training/stop'),

  getEpochResults: () => get<any>('/training/epoch-results'),

  getMLflowMetrics: (runId: string) => get<any>(`/training/metrics/${runId}`),

  listRuns: () => get<any>('/training/runs'),

  // ── Model card ────────────────────────────────────────────────────────────
  getModelCard: (runId?: string) =>
    get<any>('/model/card', runId ? { run_id: runId } : undefined),

  getClassMetrics: (forceRefresh = false) =>
    get<any>('/model/class-metrics', forceRefresh ? { force_refresh: true } : undefined),

  getPredictionDistribution: () => get<any>('/model/prediction-distribution'),

  getPoorSamples: () => get<any>('/model/poor-samples'),
}
