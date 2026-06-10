import api from './client';
import type { Team, Category } from '../types';

export const getTeams = () => api.get<Team[]>('/teams');
export const createTeam = (data: Record<string, unknown>) => api.post<Team>('/teams', data);
export const updateTeam = (id: number, data: Record<string, unknown>) => api.put<Team>(`/teams/${id}`, data);
export const deleteTeam = (id: number) => api.delete(`/teams/${id}`);

export const getCategories = () => api.get<Category[]>('/categories');
export const createCategory = (data: Record<string, unknown>) => api.post<Category>('/categories', data);
export const updateCategory = (id: number, data: Record<string, unknown>) => api.put<Category>(`/categories/${id}`, data);
export const deleteCategory = (id: number) => api.delete(`/categories/${id}`);
