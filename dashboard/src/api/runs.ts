import api from './client';
import type { PaginatedResponse, RunItem, StepResult } from '../types';

export const triggerRun = (testUuid: string, config?: Record<string, unknown>) =>
  api.post<{ run_id: string; status: string }>(`/runs/${testUuid}/trigger`, config);

export const getRuns = (params?: Record<string, unknown>) =>
  api.get<PaginatedResponse<RunItem>>('/runs', { params });

export const getRun = (runId: string) =>
  api.get<RunItem>(`/runs/${runId}`);

export const getRunSteps = (runId: string) =>
  api.get<StepResult[]>(`/runs/${runId}/steps`);

export const getTestRuns = (testUuid: string, params?: Record<string, unknown>) =>
  api.get<PaginatedResponse<RunItem>>(`/runs/test/${testUuid}/runs`, { params });

export const getTestFlakiness = (testUuid: string) =>
  api.get(`/runs/test/${testUuid}/flakiness`);

export const getReportUrl = (runId: string, format: string = 'html') =>
  `/api/runs/${runId}/report?format=${format}`;
