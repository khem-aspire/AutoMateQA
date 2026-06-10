import { useQuery } from '@tanstack/react-query';
import { getOverview, getTrends, getTeamBreakdown } from '../api/stats';
import { getRuns } from '../api/runs';
import { StatusBadge } from '../components/shared/StatusBadge';
import { LoadingSpinner } from '../components/shared/LoadingSpinner';
import {
  XAxis, YAxis, Tooltip, ResponsiveContainer, Area, AreaChart,
} from 'recharts';
import { TestTube2, Play, CheckCircle, Timer, Wrench, ArrowUpRight } from 'lucide-react';
import { Link } from 'react-router-dom';

export default function Dashboard() {
  const { data: overview, isLoading: loadingOverview, isFetching: fetchingOverview } = useQuery({
    queryKey: ['stats-overview'],
    queryFn: () => getOverview().then(r => r.data),
    staleTime: 5_000,
    refetchInterval: 10_000,
  });

  const { data: trends } = useQuery({
    queryKey: ['stats-trends'],
    queryFn: () => getTrends(30).then(r => r.data),
    staleTime: 30_000,
    refetchInterval: 30_000,
  });

  const { data: teams } = useQuery({
    queryKey: ['stats-teams'],
    queryFn: () => getTeamBreakdown().then(r => r.data),
    staleTime: 15_000,
    refetchInterval: 15_000,
  });

  const { data: recentRuns, isFetching: fetchingRuns } = useQuery({
    queryKey: ['recent-runs'],
    queryFn: () => getRuns({ per_page: 10 }).then(r => r.data),
    staleTime: 5_000,
    refetchInterval: 10_000,
  });

  const isRefreshing = fetchingOverview || fetchingRuns;

  if (loadingOverview) return <LoadingSpinner />;

  const stats = [
    {
      label: 'Total Tests',
      value: overview?.total_tests ?? 0,
      icon: TestTube2,
      accent: 'text-primary',
      accentBg: 'bg-primary/10',
      glowColor: 'shadow-primary/20',
    },
    {
      label: 'Runs Today',
      value: overview?.runs_today ?? 0,
      icon: Play,
      accent: 'text-info',
      accentBg: 'bg-info/10',
      glowColor: 'shadow-info/20',
    },
    {
      label: 'Pass Rate',
      value: `${overview?.pass_rate ?? 0}%`,
      icon: CheckCircle,
      accent: 'text-success',
      accentBg: 'bg-success/10',
      glowColor: 'shadow-success/20',
    },
    {
      label: 'Avg Duration',
      value: `${((overview?.avg_duration_ms ?? 0) / 1000).toFixed(1)}s`,
      icon: Timer,
      accent: 'text-warning',
      accentBg: 'bg-warning/10',
      glowColor: 'shadow-warning/20',
    },
    {
      label: 'Total Healed',
      value: overview?.total_healed ?? 0,
      icon: Wrench,
      accent: 'text-healed',
      accentBg: 'bg-healed/10',
      glowColor: 'shadow-healed/20',
    },
  ];

  return (
    <div className="space-y-8 animate-fade-up">
      {/* Page Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="page-title">Dashboard</h1>
          <p className="text-text-muted text-sm mt-1">Overview of your test automation health</p>
        </div>
        {isRefreshing && (
          <div className="flex items-center gap-2 text-xs text-text-muted mt-2">
            <div className="w-3 h-3 rounded-full border-2 border-primary/30 border-t-primary animate-spin" />
            Refreshing...
          </div>
        )}
      </div>

      {/* Stat Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-5 gap-4 stagger-children">
        {stats.map(({ label, value, icon: Icon, accent, accentBg, glowColor }) => (
          <div key={label} className="card-glow noise-texture p-5 group">
            <div className="relative z-10">
              <div className="flex items-center justify-between mb-4">
                <div className={`w-10 h-10 rounded-xl ${accentBg} flex items-center justify-center transition-shadow duration-300 group-hover:shadow-lg ${glowColor}`}>
                  <Icon className={`w-5 h-5 ${accent}`} />
                </div>
              </div>
              <div className="font-mono text-3xl font-bold tracking-tight">{value}</div>
              <div className="text-xs text-text-muted font-medium mt-1 uppercase tracking-wider">{label}</div>
            </div>
          </div>
        ))}
      </div>

      {/* Trend Chart */}
      {trends && trends.length > 0 && (
        <div className="card noise-texture p-6">
          <div className="relative z-10">
            <div className="flex items-center justify-between mb-6">
              <h2 className="section-title">30-Day Trend</h2>
              <div className="flex items-center gap-5 text-xs font-medium">
                <span className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-full bg-emerald-400" /> Passed
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-full bg-red-400" /> Failed
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-full bg-violet-400" /> Healed
                </span>
              </div>
            </div>
            <ResponsiveContainer width="100%" height={280}>
              <AreaChart data={trends}>
                <defs>
                  <linearGradient id="gradPassed" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#10B981" stopOpacity={0.2} />
                    <stop offset="100%" stopColor="#10B981" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="gradFailed" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#EF4444" stopOpacity={0.15} />
                    <stop offset="100%" stopColor="#EF4444" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="gradHealed" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#A78BFA" stopOpacity={0.15} />
                    <stop offset="100%" stopColor="#A78BFA" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis
                  dataKey="date"
                  stroke="#3F3F46"
                  fontSize={11}
                  fontFamily="'JetBrains Mono'"
                  tickLine={false}
                  axisLine={false}
                />
                <YAxis
                  stroke="#3F3F46"
                  fontSize={11}
                  fontFamily="'JetBrains Mono'"
                  tickLine={false}
                  axisLine={false}
                  width={32}
                />
                <Tooltip
                  contentStyle={{
                    background: '#131316',
                    border: '1px solid #27272A',
                    borderRadius: 12,
                    boxShadow: '0 8px 32px rgba(0,0,0,0.4)',
                    fontFamily: "'Plus Jakarta Sans'",
                    fontSize: 13,
                  }}
                  labelStyle={{ color: '#FAFAFA', fontWeight: 600 }}
                  itemStyle={{ color: '#A1A1AA' }}
                />
                <Area type="monotone" dataKey="passed" stroke="#10B981" strokeWidth={2} fill="url(#gradPassed)" dot={false} />
                <Area type="monotone" dataKey="failed" stroke="#EF4444" strokeWidth={2} fill="url(#gradFailed)" dot={false} />
                <Area type="monotone" dataKey="healed" stroke="#A78BFA" strokeWidth={2} fill="url(#gradHealed)" dot={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        {/* Recent Runs — 3 columns */}
        <div className="lg:col-span-3 card noise-texture p-6">
          <div className="relative z-10">
            <div className="flex justify-between items-center mb-5">
              <h2 className="section-title">Recent Runs</h2>
              <Link to="/runs" className="flex items-center gap-1 text-sm text-primary hover:text-primary-light transition-colors font-medium">
                View all <ArrowUpRight className="w-3.5 h-3.5" />
              </Link>
            </div>
            <div className="space-y-1">
              {recentRuns?.items.map((run) => (
                <Link
                  key={run.run_id}
                  to={`/runs/${run.run_id}`}
                  className="flex items-center justify-between p-3 rounded-xl hover:bg-surface-light transition-all duration-200 group"
                >
                  <div className="flex items-center gap-3 min-w-0">
                    <div className={`w-1 h-8 rounded-full ${
                      run.status === 'passed' ? 'bg-emerald-400' :
                      run.status === 'failed' ? 'bg-red-400' :
                      run.status === 'healed' ? 'bg-violet-400' :
                      run.status === 'running' ? 'bg-sky-400' :
                      'bg-zinc-500'
                    }`} />
                    <div className="min-w-0">
                      <div className="font-medium text-sm truncate group-hover:text-primary-light transition-colors">{run.test_name}</div>
                      <div className="text-xs text-text-muted">{run.team_name} &middot; {run.trigger_type}</div>
                    </div>
                  </div>
                  <div className="flex items-center gap-3 shrink-0">
                    <span className="text-xs text-text-muted font-mono">
                      {run.total_duration_ms > 0 ? `${(run.total_duration_ms / 1000).toFixed(1)}s` : '--'}
                    </span>
                    <StatusBadge status={run.status} />
                  </div>
                </Link>
              ))}
            </div>
          </div>
        </div>

        {/* Team Breakdown — 2 columns */}
        <div className="lg:col-span-2 card noise-texture p-6">
          <div className="relative z-10">
            <h2 className="section-title mb-5">Team Breakdown</h2>
            <div className="space-y-4">
              {teams?.map((team) => {
                const rateColor = team.pass_rate >= 80 ? 'text-emerald-400' : team.pass_rate >= 50 ? 'text-amber-400' : 'text-red-400';
                const barWidth = Math.max(team.pass_rate, 2);
                const barColor = team.pass_rate >= 80 ? 'bg-emerald-500' : team.pass_rate >= 50 ? 'bg-amber-500' : 'bg-red-500';
                return (
                  <div key={team.team_name}>
                    <div className="flex items-center justify-between mb-1.5">
                      <div className="flex items-center gap-2.5">
                        <span className="w-2.5 h-2.5 rounded-sm" style={{ backgroundColor: team.color }} />
                        <span className="text-sm font-medium">{team.team_name}</span>
                      </div>
                      <span className={`text-sm font-mono font-bold ${rateColor}`}>
                        {team.pass_rate}%
                      </span>
                    </div>
                    {/* Progress bar */}
                    <div className="h-1.5 bg-surface-light rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full ${barColor} transition-all duration-700 ease-out`}
                        style={{ width: `${barWidth}%` }}
                      />
                    </div>
                    <div className="flex gap-3 mt-1">
                      <span className="text-[11px] text-text-muted">{team.test_count} tests</span>
                      <span className="text-[11px] text-text-muted">{team.run_count} runs</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
