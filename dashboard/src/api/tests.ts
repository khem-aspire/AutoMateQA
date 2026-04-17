import api from './client';
import type { PaginatedResponse, TestItem, TestVersion } from '../types';

export const uploadTest = (formData: FormData) =>
  api.post<TestItem>('/tests/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });

export const getTests = (params?: Record<string, unknown>) =>
  api.get<PaginatedResponse<TestItem>>('/tests', { params });

export const getTest = (testUuid: string) =>
  api.get<TestItem>(`/tests/${testUuid}`);

export const updateTest = (testUuid: string, data: Record<string, unknown>) =>
  api.put<TestItem>(`/tests/${testUuid}`, data);

export const deleteTest = (testUuid: string) =>
  api.delete(`/tests/${testUuid}`);

export const inspectTest = (testUuid: string) =>
  api.get(`/tests/${testUuid}/inspect`);

export const getTestVersions = (testUuid: string) =>
  api.get<TestVersion[]>(`/tests/${testUuid}/versions`);

export const uploadNewVersion = (testUuid: string, file: File) => {
  const formData = new FormData();
  formData.append('file', file);
  return api.post<TestItem>(`/tests/${testUuid}/versions`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
};
