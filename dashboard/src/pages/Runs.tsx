import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { getRuns } from '../api/runs';
import { getTeams } from '../api/teams';
import { StatusBadge } from '../components/shared/StatusBadge';
import { LoadingSpinner } from '../components/shared/LoadingSpinner';
import { EmptyState } from '../components/shared/EmptyState';
import { ChevronLeft, ChevronRight } from 'lucide-react';

export default function Runs() {
  const [page, setPage] = useState(1);
  const [statusFilter, setStatusFilter] = useState('');
  const [teamFilter, setTeamFilter] = useState('');

  const { data, isLoading } = useQuery({
    queryKey: ['runs', page, statusFilter, teamFilter],
    queryFn: () => getRuns({
      page,
      per_page: 20,
      status: statusFilter || undefined,
      team_id: teamFilter || undefined,
    }).then(r => r.data),
  });

  const { data: teams } = useQuery({
    queryKey: ['teams'],
    queryFn: () => getTeams().then(r => r.data),
  });

  return (
    <div className="space-y-6 animate-fade-up">
      <div>
        <h1 className="page-title">Run History</h1>
        <p className="text-text-muted text-sm mt-1">Track and analyze test executions</p>
      </div>

      {/* Filters */}
      <div className="flex gap-3">
        <select
          value={statusFilter}
          onChange={(e) => { setStatusFilter(e.target.value); setPage(1); }}
          className="input-field w-auto min-w-[150px]"
        >
          <option value="">All Statuses</option>
          <option value="passed">Passed</option>
          <option value="failed">Failed</option>
          <option value="healed">Healed</option>
          <option value="running">Running</option>
          <option value="pending">Pending</option>
          <option value="error">Error</option>
        </select>
        <select
          value={teamFilter}
          onChange={(e) => { setTeamFilter(e.target.value); setPage(1); }}
          className="input-field w-auto min-w-[140px]"
        >
          <option value="">All Teams</option>
          {teams?.map(t => <option key={t.id} value={t.id}>{t.name}</option>)}
        </select>
      </div>

      {isLoading ? (
        <LoadingSpinner />
      ) : !data?.items.length ? (
        <EmptyState title="No runs found" description="Trigger a test run to see results here." />
      ) : (
        <>
          <div className="card overflow-hidden">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Run</th>
                  <th>Test</th>
                  <th>Team</th>
                  <th>Status</th>
                  <th>Trigger</th>
                  <th>Duration</th>
                  <th>Steps (P/F/H)</th>
                  <th>Date</th>
                </tr>
              </thead>
              <tbody>
                {data.items.map((run) => (
                  <tr key={run.run_id}>
                    <td>
                      <Link to={`/runs/${run.run_id}`} className="font-mono text-sm text-primary hover:text-primary-light transition-colors">
                        {run.run_id.slice(0, 8)}
                      </Link>
                    </td>
                    <td>
                      <Link to={`/tests/${run.test_uuid}`} className="text-sm hover:text-primary-light transition-colors">
                        {run.test_name}
                      </Link>
                    </td>
                    <td className="text-sm text-text-muted">{run.team_name}</td>
                    <td><StatusBadge status={run.status} /></td>
                    <td className="text-sm text-text-muted">{run.trigger_type}</td>
                    <td className="text-sm font-mono">{run.total_duration_ms > 0 ? `${(run.total_duration_ms / 1000).toFixed(1)}s` : '--'}</td>
                    <td className="text-sm">
                      <span className="text-emerald-400">{run.passed_count}</span>
                      <span className="text-text-muted">/</span>
                      <span className="text-red-400">{run.failed_count}</span>
                      <span className="text-text-muted">/</span>
                      <span className="text-violet-400">{run.healed_count}</span>
                    </td>
                    <td className="text-xs text-text-muted">{new Date(run.created_at).toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {data.pages > 1 && (
            <div className="flex justify-center items-center gap-1">
              <button
                onClick={() => setPage(p => Math.max(1, p - 1))}
                disabled={page === 1}
                className="p-2 rounded-lg text-text-muted hover:text-text hover:bg-surface-light disabled:opacity-30 transition-colors"
              >
                <ChevronLeft className="w-4 h-4" />
              </button>
              {Array.from({ length: data.pages }, (_, i) => i + 1).map((p) => (
                <button
                  key={p}
                  onClick={() => setPage(p)}
                  className={`w-8 h-8 rounded-lg text-sm font-medium transition-all ${
                    p === page
                      ? 'bg-primary text-white shadow-lg shadow-primary/20'
                      : 'text-text-muted hover:text-text hover:bg-surface-light'
                  }`}
                >
                  {p}
                </button>
              ))}
              <button
                onClick={() => setPage(p => Math.min(data.pages, p + 1))}
                disabled={page === data.pages}
                className="p-2 rounded-lg text-text-muted hover:text-text hover:bg-surface-light disabled:opacity-30 transition-colors"
              >
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
}
