// ── Teams ──
export interface Team {
  id: number;
  name: string;
  slug: string;
  description: string;
  color: string;
  created_at: string;
  updated_at: string;
  test_count: number;
}

// ── Categories ──
export interface Category {
  id: number;
  name: string;
  description: string;
  color: string;
  created_at: string;
  test_count: number;
}

// ── Tests ──
export interface TestItem {
  id: number;
  test_id: string;
  name: string;
  description: string;
  base_url: string;
  owner_email: string;
  team_id: number;
  team_name: string;
  category_id: number | null;
  category_name: string;
  s3_key: string;
  current_version: number;
  step_count: number;
  tags: string;
  is_active: boolean;
  created_at: string;
  updated_at: string;
  last_run_status: string | null;
}

export interface TestVersion {
  id: number;
  version: number;
  s3_key: string;
  change_reason: string;
  healed_steps: string;
  created_at: string;
}

// ── Runs ──
export interface RunItem {
  id: number;
  run_id: string;
  test_id: number;
  test_uuid: string;
  test_name: string;
  team_name: string;
  test_version: number;
  trigger_type: string;
  status: string;
  started_at: string | null;
  finished_at: string | null;
  total_duration_ms: number;
  total_steps: number;
  passed_count: number;
  failed_count: number;
  healed_count: number;
  tokens_used: number;
  error_message: string;
  created_at: string;
  result_json?: string;
}

export interface AssertionResult {
  assertion_id: string;
  assertion_type: string;
  status: string;
  message: string;
  confidence: number;
  healed: boolean;
  api_endpoint?: string;
}

export interface StepResult {
  id: number;
  step_id: number;
  action_type: string;
  status: string;
  confidence: number;
  duration_ms: number;
  retry_count: number;
  healed: boolean;
  healing_details: string;
  error: string;
}

// ── Schedules ──
export interface Schedule {
  id: number;
  test_id: number;
  test_uuid: string;
  test_name: string;
  team_name: string;
  cron_expr: string;
  timezone: string;
  enabled: boolean;
  last_run_at: string | null;
  next_run_at: string | null;
  created_at: string;
  updated_at: string;
}

// ── Stats ──
export interface OverviewStats {
  total_tests: number;
  total_runs: number;
  runs_today: number;
  pass_rate: number;
  avg_duration_ms: number;
  total_healed: number;
}

export interface TrendPoint {
  date: string;
  passed: number;
  failed: number;
  healed: number;
}

export interface CategoryBreakdown {
  category_name: string;
  color: string;
  test_count: number;
  pass_rate: number;
}

export interface TeamBreakdown {
  team_name: string;
  color: string;
  test_count: number;
  run_count: number;
  pass_rate: number;
}

// ── Pagination ──
export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  per_page: number;
  pages: number;
}

// ── WebSocket messages ──
export interface WsStepCompleted {
  type: 'step_completed';
  step_id: number;
  status: string;
  confidence: number;
  duration_ms: number;
  healed: boolean;
  healing_details: string;
  error: string;
  action_type: string;
  retry_count: number;
  total_steps: number;
}

export interface WsRunCompleted {
  type: 'run_completed' | 'run_error';
  run_id: string;
  status?: string;
  total_duration_ms?: number;
  passed?: number;
  failed?: number;
  healed?: number;
  error?: string;
}

export type WsMessage = WsStepCompleted | WsRunCompleted | { type: 'run_started' | 'heartbeat' };
