import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { getTests } from '../api/tests';
import { getTeams, getCategories } from '../api/teams';
import { StatusBadge } from '../components/shared/StatusBadge';
import { LoadingSpinner } from '../components/shared/LoadingSpinner';
import { EmptyState } from '../components/shared/EmptyState';
import { Plus, Search, ChevronLeft, ChevronRight } from 'lucide-react';

export default function Tests() {
  const [page, setPage] = useState(1);
  const [search, setSearch] = useState('');
  const [teamFilter, setTeamFilter] = useState<string>('');
  const [categoryFilter, setCategoryFilter] = useState<string>('');

  const { data, isLoading } = useQuery({
    queryKey: ['tests', page, search, teamFilter, categoryFilter],
    queryFn: () => getTests({
      page,
      per_page: 20,
      search: search || undefined,
      team_id: teamFilter || undefined,
      category_id: categoryFilter || undefined,
    }).then(r => r.data),
  });

  const { data: teams } = useQuery({
    queryKey: ['teams'],
    queryFn: () => getTeams().then(r => r.data),
  });

  const { data: categories } = useQuery({
    queryKey: ['categories'],
    queryFn: () => getCategories().then(r => r.data),
  });

  return (
    <div className="space-y-6 animate-fade-up">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="page-title">Tests</h1>
          <p className="text-text-muted text-sm mt-1">Manage your test suite</p>
        </div>
        <Link to="/tests/upload" className="btn btn-primary">
          <Plus className="w-4 h-4" /> Upload Test
        </Link>
      </div>

      {/* Filters */}
      <div className="flex gap-3 flex-wrap">
        <div className="search-input flex-1 min-w-[240px]">
          <Search className="w-4 h-4 text-text-muted shrink-0" />
          <input
            type="text"
            placeholder="Search tests..."
            value={search}
            onChange={(e) => { setSearch(e.target.value); setPage(1); }}
          />
        </div>
        <select
          value={teamFilter}
          onChange={(e) => { setTeamFilter(e.target.value); setPage(1); }}
          className="input-field w-auto min-w-[140px]"
        >
          <option value="">All Teams</option>
          {teams?.map(t => <option key={t.id} value={t.id}>{t.name}</option>)}
        </select>
        <select
          value={categoryFilter}
          onChange={(e) => { setCategoryFilter(e.target.value); setPage(1); }}
          className="input-field w-auto min-w-[160px]"
        >
          <option value="">All Categories</option>
          {categories?.map(c => <option key={c.id} value={c.id}>{c.name}</option>)}
        </select>
      </div>

      {/* Table */}
      {isLoading ? (
        <LoadingSpinner />
      ) : !data?.items.length ? (
        <EmptyState
          title="No tests found"
          description="Upload your first .aqa test file to get started."
          action={
            <Link to="/tests/upload" className="btn btn-primary">Upload Test</Link>
          }
        />
      ) : (
        <>
          <div className="card overflow-hidden">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Team</th>
                  <th>Category</th>
                  <th>Owner</th>
                  <th>Steps</th>
                  <th>Version</th>
                  <th>Last Run</th>
                </tr>
              </thead>
              <tbody>
                {data.items.map((test) => (
                  <tr key={test.test_id}>
                    <td>
                      <Link to={`/tests/${test.test_id}`} className="font-medium text-sm text-text hover:text-primary-light transition-colors">
                        {test.name}
                      </Link>
                      {test.description && (
                        <p className="text-xs text-text-muted truncate max-w-xs mt-0.5">{test.description}</p>
                      )}
                    </td>
                    <td>
                      <span className="text-xs font-semibold px-2.5 py-1 rounded-lg bg-primary/10 text-primary-light ring-1 ring-inset ring-primary/20">
                        {test.team_name}
                      </span>
                    </td>
                    <td className="text-sm text-text-muted">{test.category_name || '--'}</td>
                    <td className="text-sm text-text-muted">{test.owner_email}</td>
                    <td className="text-sm font-mono">{test.step_count}</td>
                    <td className="text-sm text-text-muted font-mono">v{test.current_version}</td>
                    <td><StatusBadge status={test.last_run_status} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
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
