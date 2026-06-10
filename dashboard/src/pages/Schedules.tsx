import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { getSchedules, createSchedule, deleteSchedule, toggleSchedule } from '../api/schedules';
import { LoadingSpinner } from '../components/shared/LoadingSpinner';
import { EmptyState } from '../components/shared/EmptyState';
import { Plus, Trash2, Search, X } from 'lucide-react';
import { Link } from 'react-router-dom';

const CRON_PRESETS = [
  { label: 'Daily at 6 AM', value: '0 6 * * *' },
  { label: 'Every 6 hours', value: '0 */6 * * *' },
  { label: 'Weekdays at 9 AM', value: '0 9 * * 1-5' },
  { label: 'Hourly', value: '0 * * * *' },
];

export default function Schedules() {
  const queryClient = useQueryClient();
  const [showModal, setShowModal] = useState(false);
  const [newTestUuid, setNewTestUuid] = useState('');
  const [newCron, setNewCron] = useState('0 6 * * *');
  const [newTimezone, setNewTimezone] = useState('UTC');
  const [error, setError] = useState('');
  const [searchQuery, setSearchQuery] = useState('');

  const { data: schedules, isLoading } = useQuery({
    queryKey: ['schedules'],
    queryFn: () => getSchedules().then(r => r.data),
  });

  const createMut = useMutation({
    mutationFn: () => createSchedule({ test_uuid: newTestUuid.trim(), cron_expr: newCron, timezone: newTimezone }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['schedules'] });
      setShowModal(false);
      setNewTestUuid('');
      setError('');
    },
    onError: (err: any) => {
      setError(err.response?.data?.detail || 'Failed to create schedule');
    },
  });

  const toggleMut = useMutation({
    mutationFn: (id: number) => toggleSchedule(id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['schedules'] }),
  });

  const deleteMut = useMutation({
    mutationFn: (id: number) => deleteSchedule(id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['schedules'] }),
  });

  const filteredSchedules = schedules?.filter(s => {
    if (!searchQuery) return true;
    const q = searchQuery.toLowerCase();
    return (
      s.test_uuid.toLowerCase().includes(q) ||
      s.test_name.toLowerCase().includes(q) ||
      s.team_name.toLowerCase().includes(q)
    );
  });

  if (isLoading) return <LoadingSpinner />;

  return (
    <div className="space-y-6 animate-fade-up">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="page-title">Schedules</h1>
          <p className="text-text-muted text-sm mt-1">Automated test execution schedules</p>
        </div>
        <button onClick={() => setShowModal(true)} className="btn btn-primary">
          <Plus className="w-4 h-4" /> Add Schedule
        </button>
      </div>

      {/* Search */}
      <div className="search-input max-w-md">
        <Search className="w-4 h-4 text-text-muted shrink-0" />
        <input
          type="text"
          placeholder="Search by test UUID, name, or team..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
        />
      </div>

      {!filteredSchedules?.length ? (
        <EmptyState
          title={searchQuery ? 'No matching schedules' : 'No schedules'}
          description={searchQuery ? 'Try a different search term.' : "Create a schedule from a test's detail page or click Add Schedule."}
        />
      ) : (
        <div className="card overflow-hidden">
          <table className="data-table">
            <thead>
              <tr>
                <th>Test</th>
                <th>UUID</th>
                <th>Team</th>
                <th>Cron</th>
                <th>Timezone</th>
                <th>Last Run</th>
                <th>Enabled</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {filteredSchedules.map((s) => (
                <tr key={s.id}>
                  <td>
                    <Link to={`/tests/${s.test_uuid}`} className="text-sm font-medium text-text hover:text-primary-light transition-colors">
                      {s.test_name}
                    </Link>
                  </td>
                  <td>
                    <span className="text-xs font-mono text-text-muted">{s.test_uuid}</span>
                  </td>
                  <td className="text-sm text-text-muted">{s.team_name}</td>
                  <td className="text-sm font-mono text-text-secondary">{s.cron_expr}</td>
                  <td className="text-sm text-text-muted">{s.timezone}</td>
                  <td className="text-xs text-text-muted">
                    {s.last_run_at ? new Date(s.last_run_at).toLocaleString() : '--'}
                  </td>
                  <td>
                    <button
                      onClick={() => toggleMut.mutate(s.id)}
                      className={`toggle-switch ${s.enabled ? 'active' : 'inactive'}`}
                    >
                      <span className="toggle-knob" />
                    </button>
                  </td>
                  <td>
                    <button
                      onClick={() => { if (window.confirm('Delete this schedule?')) deleteMut.mutate(s.id); }}
                      className="text-text-muted hover:text-red-400 transition-colors p-1.5 rounded-lg hover:bg-red-500/10"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Create Schedule Modal */}
      {showModal && (
        <div className="modal-backdrop">
          <div className="modal-content w-[480px] space-y-5">
            <div className="flex justify-between items-center">
              <h2 className="section-title">New Schedule</h2>
              <button onClick={() => { setShowModal(false); setError(''); }} className="text-text-muted hover:text-text transition-colors p-1">
                <X className="w-5 h-5" />
              </button>
            </div>

            <div>
              <label className="block text-sm font-semibold text-text mb-2">Test UUID</label>
              <input
                value={newTestUuid}
                onChange={(e) => setNewTestUuid(e.target.value)}
                className="input-field input-mono"
                placeholder="e.g. 05081d49bb0c4f2a"
              />
              <p className="text-xs text-text-muted mt-1.5">Paste the test UUID from the test detail page.</p>
            </div>

            <div>
              <label className="block text-sm font-semibold text-text mb-2">Cron Expression</label>
              <input
                value={newCron}
                onChange={(e) => setNewCron(e.target.value)}
                className="input-field input-mono"
              />
              <div className="flex flex-wrap gap-2 mt-2">
                {CRON_PRESETS.map(p => (
                  <button
                    key={p.value}
                    onClick={() => setNewCron(p.value)}
                    className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                      newCron === p.value
                        ? 'bg-primary text-white shadow-lg shadow-primary/20'
                        : 'bg-surface-light text-text-muted hover:text-text'
                    }`}
                  >
                    {p.label}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <label className="block text-sm font-semibold text-text mb-2">Timezone</label>
              <input
                value={newTimezone}
                onChange={(e) => setNewTimezone(e.target.value)}
                className="input-field"
              />
            </div>

            {error && <p className="text-sm text-danger font-medium">{error}</p>}

            <div className="flex justify-end gap-3 pt-2">
              <button onClick={() => { setShowModal(false); setError(''); }} className="btn btn-ghost">
                Cancel
              </button>
              <button
                onClick={() => createMut.mutate()}
                disabled={!newTestUuid.trim() || createMut.isPending}
                className="btn btn-primary"
              >
                {createMut.isPending ? 'Creating...' : 'Create Schedule'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
