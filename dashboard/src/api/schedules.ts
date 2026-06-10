import api from './client';
import type { Schedule } from '../types';

export const getSchedules = () =>
  api.get<Schedule[]>('/schedules');

export const createSchedule = (data: { test_uuid: string; cron_expr: string; timezone: string }) =>
  api.post<Schedule>('/schedules', data);

export const updateSchedule = (id: number, data: Record<string, unknown>) =>
  api.put<Schedule>(`/schedules/${id}`, data);

export const deleteSchedule = (id: number) =>
  api.delete(`/schedules/${id}`);

export const toggleSchedule = (id: number) =>
  api.post<Schedule>(`/schedules/${id}/toggle`);
