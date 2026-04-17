import api from './client';
import type { OverviewStats, TrendPoint, CategoryBreakdown, TeamBreakdown } from '../types';

export const getOverview = () =>
  api.get<OverviewStats>('/stats/overview');

export const getTrends = (days: number = 30) =>
  api.get<TrendPoint[]>('/stats/trends', { params: { days } });

export const getCategoryBreakdown = () =>
  api.get<CategoryBreakdown[]>('/stats/category-breakdown');

export const getTeamBreakdown = () =>
  api.get<TeamBreakdown[]>('/stats/team-breakdown');
