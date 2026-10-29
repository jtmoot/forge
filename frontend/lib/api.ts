/**
 * Typed API client for the Forge FastAPI backend.
 *
 * All functions hit the backend at NEXT_PUBLIC_API_URL (default http://localhost:8000).
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// ---------------------------------------------------------------------------
// Generic fetch helper
// ---------------------------------------------------------------------------

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    throw new Error(`API ${res.status}: ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

// ---------------------------------------------------------------------------
// Shared types (mirrors Pydantic schemas)
// ---------------------------------------------------------------------------

export interface ProjectSummary {
  name: string;
  commit_count_7d: number;
  total_cost_7d: number;
  error_count_7d: number;
  last_activity: string | null;
  health: "green" | "yellow" | "red";
}

export interface ProjectListResponse {
  projects: ProjectSummary[];
  count: number;
}

export interface OpsLogResponse {
  id: string;
  project_name: string;
  log_level: string | null;
  message: string | null;
  metadata: Record<string, unknown> | null;
  source: string | null;
  cost_usd: number | null;
  is_anomaly: boolean;
  created_at: string | null;
}

export interface GitEventResponse {
  id: string;
  repo: string;
  event_type: string | null;
  branch: string | null;
  commit_sha: string | null;
  commit_message: string | null;
  author: string | null;
  files_changed: number | null;
  additions: number | null;
  deletions: number | null;
  created_at: string | null;
}

export interface ExperimentResponse {
  id: string;
  name: string;
  description: string | null;
  dataset_id: string | null;
  status: string;
  created_at: string | null;
  updated_at: string | null;
}

export interface RunResponse {
  id: string;
  experiment_id: string;
  run_name: string | null;
  model_type: string;
  hyperparameters: Record<string, unknown>;
  feature_engineering: Record<string, unknown> | null;
  feature_set_id: string | null;
  accuracy: number | null;
  precision_score: number | null;
  recall: number | null;
  f1: number | null;
  train_loss: number | null;
  val_loss: number | null;
  test_loss: number | null;
  custom_metrics: Record<string, unknown> | null;
  inference_latency_ms: number | null;
  inference_latency_p95_ms: number | null;
  peak_memory_mb: number | null;
  model_size_mb: number | null;
  throughput_samples_per_sec: number | null;
  training_time_seconds: number | null;
  efficiency_score: number | null;
  data_version_hash: string | null;
  wandb_run_id: string | null;
  s3_artifact_path: string | null;
  status: string;
  started_at: string | null;
  completed_at: string | null;
  created_at: string | null;
}

export interface ExperimentDetailResponse extends ExperimentResponse {
  runs: RunResponse[];
}

export interface ExperimentListResponse {
  experiments: ExperimentResponse[];
  count: number;
}

export interface IntermediateResult {
  tool: string;
  result_preview: string;
}

export interface AgentQueryResponse {
  answer: string;
  tools_used: string[];
  intermediate_results: IntermediateResult[];
}

export interface ProjectDetailResponse {
  name: string;
  recent_logs: OpsLogResponse[];
  git_events: GitEventResponse[];
  linked_experiments: ExperimentResponse[];
}

export interface ActivityFeedItem {
  type: "git_commit" | "ops_log" | "experiment_completion";
  timestamp: string;
  project: string | null;
  summary: string;
  detail: Record<string, unknown> | null;
}

export interface ActivityFeedResponse {
  items: ActivityFeedItem[];
  count: number;
}

export interface DashboardSummaryResponse {
  total_projects: number;
  active_experiments: number;
  ops_alerts_24h: number;
  weekly_llm_cost: number;
  drift_alerts_7d: number;
}

// ---------------------------------------------------------------------------
// API functions
// ---------------------------------------------------------------------------

export function fetchProjects(): Promise<ProjectListResponse> {
  return apiFetch<ProjectListResponse>("/api/projects");
}

export function fetchProjectDetail(name: string): Promise<ProjectDetailResponse> {
  return apiFetch<ProjectDetailResponse>(`/api/projects/${encodeURIComponent(name)}`);
}

export function fetchActivityFeed(limit = 20): Promise<ActivityFeedResponse> {
  return apiFetch<ActivityFeedResponse>(`/api/activity/feed?limit=${limit}`);
}

export function fetchDashboardSummary(): Promise<DashboardSummaryResponse> {
  return apiFetch<DashboardSummaryResponse>("/api/dashboard/summary");
}

// ---------------------------------------------------------------------------
// Metrics Summary (Prometheus-backed)
// ---------------------------------------------------------------------------

export interface MetricsSummaryResponse {
  total_requests: number;
  total_errors: number;
  error_rate_pct: number;
  experiments_total: number;
  llm_cost_dollars: number;
}

export function fetchMetricsSummary(): Promise<MetricsSummaryResponse> {
  return apiFetch<MetricsSummaryResponse>("/api/metrics/summary");
}

export function fetchExperiments(): Promise<ExperimentListResponse> {
  return apiFetch<ExperimentListResponse>("/api/experiments");
}

export function fetchExperimentDetail(id: string): Promise<ExperimentDetailResponse> {
  return apiFetch<ExperimentDetailResponse>(`/api/experiments/${encodeURIComponent(id)}`);
}

export async function sendAgentQuery(question: string): Promise<AgentQueryResponse> {
  // Route through the Next.js server-side proxy at /api/agent which attaches
  // the AGENT_API_KEY Bearer token. This keeps the secret off the client.
  const res = await fetch("/api/agent", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `API ${res.status}: ${res.statusText}`);
  }
  return res.json() as Promise<AgentQueryResponse>;
}

// ---------------------------------------------------------------------------
// Feature Store
// ---------------------------------------------------------------------------

export interface FeatureRegistryEntry {
  id: string;
  feature_set_id: string;
  dataset_id: string;
  storage_path: string | null;
  row_count: number | null;
  computed_at: string | null;
  status: string;
}

export interface FeatureSetResponse {
  id: string;
  name: string;
  version: number;
  description: string | null;
  feature_config: Record<string, unknown>;
  feature_columns: string[] | null;
  created_at: string | null;
  is_active: string | null;
}

export interface FeatureSetDetailResponse extends FeatureSetResponse {
  registry_entries: FeatureRegistryEntry[];
}

export interface FeatureSetListResponse {
  feature_sets: FeatureSetResponse[];
  count: number;
}

export interface FeatureSetCompareResponse {
  feature_set_a: Record<string, unknown>;
  feature_set_b: Record<string, unknown>;
  columns_added: string[];
  columns_removed: string[];
  config_added: Record<string, unknown>;
  config_removed: Record<string, unknown>;
  config_changed: Record<string, unknown>;
}

export function fetchFeatureSets(): Promise<FeatureSetListResponse> {
  return apiFetch<FeatureSetListResponse>("/api/features");
}

export function fetchFeatureSetDetail(id: string): Promise<FeatureSetDetailResponse> {
  return apiFetch<FeatureSetDetailResponse>(`/api/features/${encodeURIComponent(id)}`);
}

export function compareFeatureSets(a: string, b: string): Promise<FeatureSetCompareResponse> {
  return apiFetch<FeatureSetCompareResponse>(`/api/features/compare?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}`);
}

// ---------------------------------------------------------------------------
// Model Registry
// ---------------------------------------------------------------------------

export interface ModelVersionResponse {
  id: string;
  model_id: string;
  version: number;
  run_id: string;
  stage: string;
  stage_changed_at: string | null;
  stage_changed_by: string | null;
  s3_artifact_path: string | null;
  model_size_mb: number | null;
  metrics_snapshot: Record<string, number | null> | null;
  tags: Record<string, unknown> | null;
  created_at: string | null;
}

export interface ModelStageHistoryEntry {
  id: string;
  model_version_id: string;
  from_stage: string;
  to_stage: string;
  changed_at: string | null;
  reason: string | null;
}

export interface ModelListItem {
  id: string;
  name: string;
  description: string | null;
  version_count: number;
  production_version: number | null;
  production_accuracy: number | null;
  created_at: string | null;
  updated_at: string | null;
}

export interface ModelListResponse {
  models: ModelListItem[];
  count: number;
}

export interface ModelDetailResponse {
  id: string;
  name: string;
  description: string | null;
  created_at: string | null;
  updated_at: string | null;
  versions: ModelVersionResponse[];
  stage_history: ModelStageHistoryEntry[];
}

export interface ModelVersionCompareResponse {
  version_a: Record<string, unknown>;
  version_b: Record<string, unknown>;
  metrics: Record<string, { version_a: number | null; version_b: number | null; delta?: number; pct_change?: number | null }>;
}

export function fetchModels(): Promise<ModelListResponse> {
  return apiFetch<ModelListResponse>("/api/models");
}

export function fetchModelDetail(name: string): Promise<ModelDetailResponse> {
  return apiFetch<ModelDetailResponse>(`/api/models/${encodeURIComponent(name)}`);
}

export function transitionModelStage(name: string, version: number, stage: string, reason?: string): Promise<ModelVersionResponse> {
  return apiFetch<ModelVersionResponse>(`/api/models/${encodeURIComponent(name)}/versions/${version}/stage`, {
    method: "PATCH",
    body: JSON.stringify({ stage, reason }),
  });
}

export function registerModelVersion(name: string, runId: string, tags?: Record<string, unknown>): Promise<ModelVersionResponse> {
  return apiFetch<ModelVersionResponse>(`/api/models/${encodeURIComponent(name)}/versions`, {
    method: "POST",
    body: JSON.stringify({ run_id: runId, tags }),
  });
}

export function compareModelVersions(name: string, a: number, b: number): Promise<ModelVersionCompareResponse> {
  return apiFetch<ModelVersionCompareResponse>(`/api/models/${encodeURIComponent(name)}/compare?a=${a}&b=${b}`);
}

// ---------------------------------------------------------------------------
// Drift Detection
// ---------------------------------------------------------------------------

export interface DriftReportResponse {
  id: string;
  dataset_id: string;
  reference_dataset_id: string;
  report_type: string;
  model_version_id: string | null;
  overall_drift_score: number | null;
  is_drifted: string | null;
  feature_scores: Record<string, unknown> | null;
  config: Record<string, unknown> | null;
  created_at: string | null;
}

export interface DriftReportListResponse {
  reports: DriftReportResponse[];
  count: number;
}

export interface DriftSummaryResponse {
  total_reports: number;
  drifted_count: number;
  datasets_with_drift: number;
  by_type: Record<string, number>;
  last_check: string | null;
  days: number;
}

export function fetchDriftReports(params?: { dataset_id?: string; report_type?: string; is_drifted?: boolean }): Promise<DriftReportListResponse> {
  const searchParams = new URLSearchParams();
  if (params?.dataset_id) searchParams.set("dataset_id", params.dataset_id);
  if (params?.report_type) searchParams.set("report_type", params.report_type);
  if (params?.is_drifted !== undefined) searchParams.set("is_drifted", String(params.is_drifted));
  const qs = searchParams.toString();
  return apiFetch<DriftReportListResponse>(`/api/drift/reports${qs ? "?" + qs : ""}`);
}

export function fetchDriftReport(id: string): Promise<DriftReportResponse> {
  return apiFetch<DriftReportResponse>(`/api/drift/reports/${encodeURIComponent(id)}`);
}

export function fetchDriftSummary(days: number = 30): Promise<DriftSummaryResponse> {
  return apiFetch<DriftSummaryResponse>(`/api/drift/summary?days=${days}`);
}

// ---------------------------------------------------------------------------
// Reproducibility
// ---------------------------------------------------------------------------

export interface RunEnvironmentResponse {
  id: string;
  run_id: string;
  git_sha: string | null;
  git_branch: string | null;
  git_dirty: boolean | null;
  python_version: string | null;
  package_versions: Record<string, string> | null;
  docker_image_tag: string | null;
  random_seed: number | null;
  env_hash: string | null;
  created_at: string | null;
}

export interface ReproduceResponse {
  git_sha: string | null;
  command: string;
  data_version: string | null;
  feature_set: string | null;
  environment_hash: string | null;
  random_seed: number | null;
  warnings: string[];
}

export interface ReproducibilityReport {
  verdict: string;
  factors: Record<string, { run_a: unknown; run_b: unknown; match: boolean }>;
  warnings: string[];
}

export interface EnvironmentDiffResponse {
  run_a: RunEnvironmentResponse | null;
  run_b: RunEnvironmentResponse | null;
  packages_added: Record<string, string>;
  packages_removed: Record<string, string>;
  packages_changed: Record<string, { run_a: string; run_b: string }>;
  field_diffs: Record<string, { run_a: unknown; run_b: unknown }>;
  environments_identical: boolean;
  reproducibility: ReproducibilityReport | null;
}

export function fetchRunEnvironment(experimentId: string, runId: string): Promise<RunEnvironmentResponse> {
  return apiFetch<RunEnvironmentResponse>(`/api/experiments/${encodeURIComponent(experimentId)}/runs/${encodeURIComponent(runId)}/environment`);
}

export function fetchReproduceSpec(experimentId: string, runId: string): Promise<ReproduceResponse> {
  return apiFetch<ReproduceResponse>(`/api/experiments/${encodeURIComponent(experimentId)}/runs/${encodeURIComponent(runId)}/reproduce`);
}

export function compareEnvironments(runA: string, runB: string): Promise<EnvironmentDiffResponse> {
  return apiFetch<EnvironmentDiffResponse>(`/api/experiments/compare-environments?run_a=${encodeURIComponent(runA)}&run_b=${encodeURIComponent(runB)}`);
}

// ---------------------------------------------------------------------------
// Serving (mirrors forge/api/models/schemas.py serving schemas)
// ---------------------------------------------------------------------------

export interface BackendParity {
  backend: string;
  passed: boolean;
  skipped: boolean;
  max_abs_diff: number | null;
  label_agreement: number | null;
  max_abs_diff_limit: number | null;
  min_label_agreement: number | null;
  samples: number;
  reason: string | null;
}

export interface BatchConfigResponse {
  model_name: string;
  max_batch_size: number;
  max_wait_ms: number;
  enabled: boolean;
}

export interface ServedModelResponse {
  model_name: string;
  version: number;
  run_id: string;
  model_type: string;
  backend: string;
  backends: string[];
  parity: BackendParity[];
  input_shape: number[];
  feature_names: string[];
  loaded_at: string;
  load_seconds: number;
  onnx_build_seconds: number;
  batching: BatchConfigResponse;
}

export interface BenchmarkCell {
  backend: string;
  max_batch_size: number;
  max_wait_ms: number;
  concurrency: number;
  requests: number;
  errors: number;
  error_rate: number;
  p50_ms: number | null;
  p95_ms: number | null;
  p99_ms: number | null;
  mean_ms: number | null;
  throughput_rps: number;
  mean_batch_rows: number;
  duration_s: number;
}

export interface BenchmarkResponse {
  id: string;
  model_name: string;
  model_version: number;
  status: "running" | "completed" | "failed";
  params: {
    backends: string[] | null;
    batch_configs: { max_batch_size: number; max_wait_ms: number }[];
    concurrency: number[];
    requests_per_cell: number;
    latency_scope?: string;
    input_samples?: number;
  } | null;
  runtime: { cpu_count: number; cpu_limit: number | null; torch_threads: number } | null;
  results: BenchmarkCell[];
  error: string | null;
  started_at: string | null;
  completed_at: string | null;
}

export interface BenchmarkSummary {
  id: string;
  model_name: string;
  model_version: number;
  status: string;
  cells: number;
  started_at: string | null;
  completed_at: string | null;
}

export function fetchServedModel(name: string): Promise<ServedModelResponse> {
  return apiFetch<ServedModelResponse>(`/api/serve/${encodeURIComponent(name)}`);
}

export function fetchBenchmarks(name: string, limit = 10): Promise<{ benchmarks: BenchmarkSummary[] }> {
  return apiFetch<{ benchmarks: BenchmarkSummary[] }>(`/api/serve/${encodeURIComponent(name)}/benchmarks?limit=${limit}`);
}

export function fetchBenchmark(id: string): Promise<BenchmarkResponse> {
  return apiFetch<BenchmarkResponse>(`/api/serve/benchmarks/${encodeURIComponent(id)}`);
}

export async function startBenchmark(name: string): Promise<BenchmarkResponse> {
  const res = await fetch(`${API_BASE}/api/serve/${encodeURIComponent(name)}/benchmarks`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail ?? `API ${res.status}: ${res.statusText}`);
  }
  return res.json() as Promise<BenchmarkResponse>;
}
