import { useState, useRef } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { getTest, getTestVersions, deleteTest, uploadNewVersion } from '../api/tests';
import { getTestFlakiness, triggerRun, getRuns } from '../api/runs';
import { createSchedule } from '../api/schedules';
import { StatusBadge } from '../components/shared/StatusBadge';
import { LoadingSpinner } from '../components/shared/LoadingSpinner';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
} from 'recharts';
import { Play, Download, Trash2, Clock, GitBranch, X, Upload, ArrowLeft } from 'lucide-react';

const CRON_PRESETS = [
  { label: 'Daily at 6 AM', value: '0 6 * * *' },
  { label: 'Every 6 hours', value: '0 */6 * * *' },
  { label: 'Weekdays at 9 AM', value: '0 9 * * 1-5' },
  { label: 'Hourly', value: '0 * * * *' },
];

export default function TestDetail() {
  const { testUuid } = useParams<{ testUuid: string }>();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const versionFileRef = useRef<HTMLInputElement>(null);
  const [tab, setTab] = useState<'overview' | 'history' | 'flakiness' | 'versions'>('overview');
  const [showScheduleModal, setShowScheduleModal] = useState(false);
  const [cronExpr, setCronExpr] = useState('0 6 * * *');
  const [timezone, setTimezone] = useState('UTC');
  const [scheduleError, setScheduleError] = useState('');
  const [versionError, setVersionError] = useState('');

  const uuid = testUuid!;

  const { data: test, isLoading } = useQuery({
    queryKey: ['test', uuid],
    queryFn: () => getTest(uuid).then(r => r.data),
  });

  const { data: runs } = useQuery({
    queryKey: ['test-runs', uuid],
    queryFn: () => getRuns({ test_uuid: uuid, per_page: 20 }).then(r => r.data),
    enabled: tab === 'history' || tab === 'overview',
  });

  const { data: flakiness } = useQuery({
    queryKey: ['test-flakiness', uuid],
    queryFn: () => getTestFlakiness(uuid).then(r => r.data),
    enabled: tab === 'flakiness',
  });

  const { data: versions } = useQuery({
    queryKey: ['test-versions', uuid],
    queryFn: () => getTestVersions(uuid).then(r => r.data),
    enabled: tab === 'versions',
  });

  const runMutation = useMutation({
    mutationFn: () => triggerRun(uuid),
    onSuccess: (res) => navigate(`/runs/${res.data.run_id}`),
  });

  const deleteMutation = useMutation({
    mutationFn: () => deleteTest(uuid),
    onSuccess: () => navigate('/tests'),
  });

  const versionMutation = useMutation({
    mutationFn: (file: File) => uploadNewVersion(uuid, file),
    onSuccess: () => {
      setVersionError('');
      queryClient.invalidateQueries({ queryKey: ['test', uuid] });
      queryClient.invalidateQueries({ queryKey: ['test-versions', uuid] });
    },
    onError: (err: any) => {
      setVersionError(err.response?.data?.detail || 'Failed to upload new version');
    },
  });

  const handleVersionUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setVersionError('');
      versionMutation.mutate(file);
    }
    if (versionFileRef.current) versionFileRef.current.value = '';
  };

  const scheduleMutation = useMutation({
    mutationFn: () => createSchedule({ test_uuid: uuid, cron_expr: cronExpr, timezone }),
    onSuccess: () => {
      setShowScheduleModal(false);
      setScheduleError('');
    },
    onError: (err: any) => {
      setScheduleError(err.response?.data?.detail || 'Failed to create schedule');
    },
  });

  if (isLoading) return <LoadingSpinner />;
  if (!test) return <div className="text-text-muted text-center py-20">Test not found</div>;

  const tabs = [
    { key: 'overview', label: 'Overview' },
    { key: 'history', label: 'Run History' },
    { key: 'flakiness', label: 'Flakiness' },
    { key: 'versions', label: 'Versions' },
  ] as const;

  return (
    <div className="space-y-6 animate-fade-up">
      {/* Breadcrumb */}
      <Link to="/tests" className="inline-flex items-center gap-1.5 text-sm text-text-muted hover:text-text transition-colors">
        <ArrowLeft className="w-4 h-4" /> Back to Tests
      </Link>

      {/* Header */}
      <div className="flex justify-between items-start gap-4">
        <div className="min-w-0">
          <h1 className="page-title truncate">{test.name}</h1>
          <p className="text-sm text-text-muted mt-1">{test.description || 'No description'}</p>
          <div className="flex items-center gap-2 mt-3 flex-wrap">
            <span className="font-mono text-xs px-2.5 py-1 rounded-lg bg-surface-light text-text-muted">{test.test_id}</span>
            <span className="text-xs font-semibold px-2.5 py-1 rounded-lg bg-primary/10 text-primary-light ring-1 ring-inset ring-primary/20">{test.team_name}</span>
            {test.category_name && <span className="text-xs px-2.5 py-1 rounded-lg bg-surface-light text-text-secondary">{test.category_name}</span>}
            <span className="text-xs text-text-muted">{test.owner_email}</span>
            <span className="text-xs font-mono text-text-muted">v{test.current_version}</span>
            <span className="text-xs text-text-muted">{test.step_count} steps</span>
          </div>
        </div>
        <div className="flex gap-2 shrink-0">
          <button onClick={() => runMutation.mutate()} disabled={runMutation.isPending} className="btn btn-success">
            <Play className="w-4 h-4" /> Run Now
          </button>
          <button onClick={() => setShowScheduleModal(true)} className="btn btn-ghost">
            <Clock className="w-4 h-4" /> Schedule
          </button>
          <a href={`/api/tests/${uuid}/download`} className="btn btn-ghost">
            <Download className="w-4 h-4" />
          </a>
          <button
            onClick={() => { if (window.confirm('Delete this test and all its runs?')) deleteMutation.mutate(); }}
            className="btn btn-danger"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 border-b border-border">
        {tabs.map(({ key, label }) => (
          <button
            key={key}
            onClick={() => setTab(key)}
            className={`px-5 py-3 text-sm font-medium border-b-2 transition-all ${
              tab === key
                ? 'border-primary text-primary-light'
                : 'border-transparent text-text-muted hover:text-text'
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      {tab === 'overview' && (
        <div className="grid grid-cols-2 gap-4 stagger-children">
          <div className="card p-5">
            <h3 className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-2">Base URL</h3>
            <p className="text-sm font-mono text-text">{test.base_url || '--'}</p>
          </div>
          <div className="card p-5">
            <h3 className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-2">Last Run</h3>
            <StatusBadge status={test.last_run_status} />
          </div>
          <div className="card p-5">
            <h3 className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-2">Tags</h3>
            <p className="text-sm">{test.tags || 'None'}</p>
          </div>
          <div className="card p-5">
            <h3 className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-2">Created</h3>
            <p className="text-sm">{new Date(test.created_at).toLocaleDateString()}</p>
          </div>
        </div>
      )}

      {tab === 'history' && (
        <div className="card overflow-hidden">
          <table className="data-table">
            <thead>
              <tr>
                <th>Run ID</th>
                <th>Status</th>
                <th>Trigger</th>
                <th>Duration</th>
                <th>Steps</th>
                <th>Healed</th>
                <th>Date</th>
              </tr>
            </thead>
            <tbody>
              {runs?.items.map((run) => (
                <tr key={run.run_id}>
                  <td>
                    <Link to={`/runs/${run.run_id}`} className="text-sm text-primary hover:text-primary-light font-mono transition-colors">
                      {run.run_id.slice(0, 8)}
                    </Link>
                  </td>
                  <td><StatusBadge status={run.status} /></td>
                  <td className="text-sm text-text-muted">{run.trigger_type}</td>
                  <td className="text-sm font-mono">{(run.total_duration_ms / 1000).toFixed(1)}s</td>
                  <td className="text-sm">
                    <span className="text-emerald-400">{run.passed_count}</span>
                    <span className="text-text-muted">/</span>
                    <span className="text-red-400">{run.failed_count}</span>
                    <span className="text-text-muted">/</span>
                    <span className="text-violet-400">{run.healed_count}</span>
                  </td>
                  <td className="text-sm text-violet-400 font-mono">{run.healed_count}</td>
                  <td className="text-xs text-text-muted">{new Date(run.created_at).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {tab === 'flakiness' && flakiness && (
        <div className="card p-6">
          <h2 className="section-title mb-5">Per-Step Flakiness Score</h2>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={flakiness}>
              <XAxis
                dataKey="step_id"
                stroke="#3F3F46"
                fontSize={11}
                fontFamily="'JetBrains Mono'"
                tickLine={false}
                axisLine={false}
                label={{ value: 'Step', position: 'insideBottom', offset: -5, style: { fill: '#71717A', fontSize: 11 } }}
              />
              <YAxis
                stroke="#3F3F46"
                fontSize={11}
                fontFamily="'JetBrains Mono'"
                tickLine={false}
                axisLine={false}
                domain={[0, 1]}
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
              />
              <Bar dataKey="flakiness_score" fill="#A78BFA" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {tab === 'versions' && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="section-title">Version History</h2>
            <div className="flex items-center gap-3">
              {versionMutation.isPending && (
                <span className="text-sm text-text-muted flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full border-2 border-primary/30 border-t-primary animate-spin" />
                  Uploading...
                </span>
              )}
              <input ref={versionFileRef} type="file" accept=".aqa,.json" onChange={handleVersionUpload} className="hidden" />
              <button onClick={() => versionFileRef.current?.click()} disabled={versionMutation.isPending} className="btn btn-primary">
                <Upload className="w-4 h-4" /> Upload New Version
              </button>
            </div>
          </div>

          {versionError && (
            <div className="px-4 py-3 bg-danger/10 border border-danger/20 rounded-xl text-sm text-danger font-medium">
              {versionError}
            </div>
          )}

          <div className="space-y-3 stagger-children">
            {versions?.map((v) => (
              <div key={v.id} className="card-glow flex items-center gap-4 p-5">
                <div className="w-11 h-11 rounded-xl bg-primary/10 flex items-center justify-center shrink-0">
                  <GitBranch className="w-5 h-5 text-primary" />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="font-semibold text-sm">Version {v.version}</span>
                    <span className={`text-xs font-semibold px-2 py-0.5 rounded-md ${
                      v.change_reason === 'self_healed'
                        ? 'bg-violet-500/15 text-violet-400 ring-1 ring-inset ring-violet-500/20'
                        : v.change_reason === 'manual_edit'
                          ? 'bg-sky-500/15 text-sky-400 ring-1 ring-inset ring-sky-500/20'
                          : 'bg-zinc-500/15 text-zinc-400 ring-1 ring-inset ring-zinc-500/20'
                    }`}>
                      {v.change_reason === 'manual_edit' ? 'manual upload' : v.change_reason}
                    </span>
                    {v.version === test.current_version && (
                      <span className="text-xs font-semibold px-2 py-0.5 rounded-md bg-emerald-500/15 text-emerald-400 ring-1 ring-inset ring-emerald-500/20">current</span>
                    )}
                  </div>
                  <p className="text-xs text-text-muted mt-1">
                    {new Date(v.created_at).toLocaleString()}
                    {v.healed_steps && v.healed_steps !== '[]' && ` -- Healed steps: ${v.healed_steps}`}
                  </p>
                </div>
                <a href={`/api/tests/${uuid}/download?version=${v.version}`} className="text-text-muted hover:text-primary transition-colors p-2">
                  <Download className="w-4 h-4" />
                </a>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Schedule Modal */}
      {showScheduleModal && (
        <div className="modal-backdrop">
          <div className="modal-content w-[480px] space-y-5">
            <div className="flex justify-between items-center">
              <h2 className="section-title">Schedule: {test.name}</h2>
              <button onClick={() => { setShowScheduleModal(false); setScheduleError(''); }} className="text-text-muted hover:text-text transition-colors p-1">
                <X className="w-5 h-5" />
              </button>
            </div>
            <p className="text-xs text-text-muted font-mono">{test.test_id}</p>

            <div>
              <label className="block text-sm font-semibold text-text mb-2">Cron Expression</label>
              <input
                value={cronExpr}
                onChange={(e) => setCronExpr(e.target.value)}
                className="input-field input-mono"
              />
              <div className="flex flex-wrap gap-2 mt-2">
                {CRON_PRESETS.map(p => (
                  <button
                    key={p.value}
                    onClick={() => setCronExpr(p.value)}
                    className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                      cronExpr === p.value
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
              <input value={timezone} onChange={(e) => setTimezone(e.target.value)} className="input-field" />
            </div>

            {scheduleError && <p className="text-sm text-danger font-medium">{scheduleError}</p>}

            <div className="flex justify-end gap-3 pt-2">
              <button onClick={() => { setShowScheduleModal(false); setScheduleError(''); }} className="btn btn-ghost">
                Cancel
              </button>
              <button onClick={() => scheduleMutation.mutate()} disabled={scheduleMutation.isPending} className="btn btn-primary">
                {scheduleMutation.isPending ? 'Creating...' : 'Create Schedule'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
