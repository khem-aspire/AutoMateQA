import { useParams, Link } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { getRun, getRunSteps, getReportUrl } from '../api/runs';
import { useWebSocket } from '../hooks/useWebSocket';
import { StatusBadge } from '../components/shared/StatusBadge';
import { LoadingSpinner } from '../components/shared/LoadingSpinner';
import {
  ChevronDown, ChevronRight, Download, FileText, FileJson, FileCode,
  CheckCircle2, XCircle, ShieldCheck, ArrowLeft,
} from 'lucide-react';
import { useState, useMemo } from 'react';
import type { StepResult, AssertionResult, WsStepCompleted } from '../types';

function parseAssertions(resultJson: string | undefined): Record<number, AssertionResult[]> {
  if (!resultJson) return {};
  try {
    const result = JSON.parse(resultJson);
    const map: Record<number, AssertionResult[]> = {};
    for (const step of result.steps || []) {
      if (step.assertions && step.assertions.length > 0) {
        map[step.step_id] = step.assertions;
      }
    }
    return map;
  } catch {
    return {};
  }
}

function AssertionRow({ assertion }: { assertion: AssertionResult }) {
  const passed = assertion.status === 'passed';
  return (
    <div className={`flex items-start gap-3 px-4 py-3 rounded-xl ${
      passed ? 'bg-emerald-500/5 ring-1 ring-inset ring-emerald-500/10' : 'bg-red-500/5 ring-1 ring-inset ring-red-500/10'
    }`}>
      <div className="mt-0.5">
        {passed
          ? <CheckCircle2 className="w-4 h-4 text-emerald-400" />
          : <XCircle className="w-4 h-4 text-red-400" />
        }
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 flex-wrap">
          <span className={`text-xs font-bold uppercase tracking-wide ${passed ? 'text-emerald-400' : 'text-red-400'}`}>
            {assertion.assertion_type.replace(/_/g, ' ')}
          </span>
          {assertion.healed && (
            <span className="text-xs font-semibold px-1.5 py-0.5 rounded-md bg-violet-500/15 text-violet-400 ring-1 ring-inset ring-violet-500/20">healed</span>
          )}
          {assertion.confidence > 0 && (
            <span className="text-xs text-text-muted font-mono">conf: {assertion.confidence.toFixed(2)}</span>
          )}
        </div>
        <p className="text-sm text-text-muted mt-0.5">{assertion.message}</p>
      </div>
    </div>
  );
}

function StepRow({
  step,
  assertions,
  expanded,
  onToggle,
}: {
  step: StepResult;
  assertions: AssertionResult[];
  expanded: boolean;
  onToggle: () => void;
}) {
  const confidenceColor = step.confidence >= 0.8 ? 'text-emerald-400' : step.confidence >= 0.5 ? 'text-amber-400' : 'text-red-400';
  const assertionCount = assertions.length;
  const assertionsPassed = assertions.filter(a => a.status === 'passed').length;
  const assertionsFailed = assertionCount - assertionsPassed;

  return (
    <>
      <tr onClick={onToggle} className="cursor-pointer">
        <td className="text-sm">
          <span className="inline-flex items-center gap-1">
            {expanded ? <ChevronDown className="w-4 h-4 text-text-muted" /> : <ChevronRight className="w-4 h-4 text-text-muted" />}
            Step {step.step_id}
          </span>
        </td>
        <td className="text-sm">{step.action_type}</td>
        <td><StatusBadge status={step.status} /></td>
        <td className={`text-sm font-mono font-semibold ${confidenceColor}`}>{step.confidence.toFixed(2)}</td>
        <td className="text-sm font-mono">{step.duration_ms.toFixed(0)}ms</td>
        <td className="text-sm text-text-muted">{step.retry_count}</td>
        <td className="text-sm">
          {assertionCount > 0 && (
            <span className="flex items-center gap-1.5">
              <ShieldCheck className="w-3.5 h-3.5 text-text-muted" />
              <span className="text-emerald-400">{assertionsPassed}</span>
              {assertionsFailed > 0 && (
                <>/<span className="text-red-400">{assertionsFailed}</span></>
              )}
            </span>
          )}
        </td>
      </tr>
      {expanded && (
        <tr>
          <td colSpan={7} className="!p-0">
            <div className="px-6 py-5 bg-bg/60 border-y border-border space-y-3">
              {assertions.length > 0 && (
                <div className="space-y-2">
                  <h4 className="text-xs font-bold text-text-muted uppercase tracking-wider flex items-center gap-1.5">
                    <ShieldCheck className="w-3.5 h-3.5" />
                    Assertions ({assertionsPassed}/{assertionCount} passed)
                  </h4>
                  {assertions.map((a) => (
                    <AssertionRow key={a.assertion_id} assertion={a} />
                  ))}
                </div>
              )}

              {step.healed && (
                <div className="px-4 py-3 bg-violet-500/5 ring-1 ring-inset ring-violet-500/15 rounded-xl">
                  <span className="font-semibold text-violet-400 text-sm">Healed: </span>
                  <span className="text-text-muted text-sm">{step.healing_details}</span>
                </div>
              )}

              {step.error && (
                <div className="px-4 py-3 bg-red-500/5 ring-1 ring-inset ring-red-500/15 rounded-xl">
                  <span className="font-semibold text-red-400 text-sm">Error: </span>
                  <span className="text-text-muted text-sm">{step.error}</span>
                </div>
              )}

              {!assertions.length && !step.healed && !step.error && (
                <p className="text-text-muted text-sm">No assertions or additional details for this step.</p>
              )}
            </div>
          </td>
        </tr>
      )}
    </>
  );
}

function LivePanel({ runId, totalSteps }: { runId: string; totalSteps: number }) {
  const { messages, connected } = useWebSocket(runId);
  const steps = messages.filter(m => m.type === 'step_completed') as WsStepCompleted[];
  const started = messages.some(m => m.type === 'run_started');
  const completed = messages.find(m => m.type === 'run_completed' || m.type === 'run_error');
  const stepsTotal = steps.length > 0 ? (steps[0].total_steps || totalSteps) : totalSteps;
  const nextStep = steps.length + 1;
  const progress = stepsTotal > 0 ? (steps.length / stepsTotal) * 100 : 0;

  return (
    <div className="card noise-texture p-6 ring-1 ring-inset ring-primary/20">
      <div className="relative z-10">
        <div className="flex items-center justify-between mb-5">
          <div className="flex items-center gap-3">
            <div className="relative">
              <span className={`flex h-3 w-3 ${connected ? '' : ''}`}>
                {connected && <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />}
                <span className={`relative inline-flex rounded-full h-3 w-3 ${connected ? 'bg-emerald-400' : 'bg-red-400'}`} />
              </span>
            </div>
            <h2 className="section-title">Live Execution</h2>
          </div>
          {!completed && stepsTotal > 0 && (
            <span className="text-xs text-text-muted font-mono">
              {steps.length}/{stepsTotal} steps
            </span>
          )}
        </div>

        {/* Progress bar */}
        {!completed && stepsTotal > 0 && (
          <div className="w-full h-2 bg-surface-light rounded-full mb-5 overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-primary to-primary-light rounded-full transition-all duration-700 ease-out"
              style={{ width: `${progress}%` }}
            />
          </div>
        )}

        <div className="space-y-2">
          {steps.length === 0 && !completed && (
            <div className="flex items-center gap-3 text-sm py-4">
              <div className="w-5 h-5 rounded-full border-2 border-primary/30 border-t-primary animate-spin" />
              <span className="text-text-muted">
                {!connected
                  ? 'Connecting to execution stream...'
                  : !started
                    ? 'Waiting for execution to begin...'
                    : 'Launching browser and navigating to test URL...'
                }
              </span>
            </div>
          )}

          {steps.length > 0 && !completed && nextStep <= stepsTotal && (
            <div className="flex items-center gap-3 text-sm px-4 py-3 rounded-xl bg-primary/5 ring-1 ring-inset ring-primary/15 animate-pulse-glow">
              <div className="w-4 h-4 rounded-full border-2 border-primary/30 border-t-primary animate-spin" />
              <span className="text-primary-light font-medium">Executing Step {nextStep}...</span>
            </div>
          )}

          {steps.map((s) => (
            <div key={s.step_id} className="flex items-center gap-3 text-sm px-3 py-2 rounded-lg animate-slide-in">
              <StatusBadge status={s.status} />
              <span className="font-medium">Step {s.step_id}: <span className="text-text-secondary">{s.action_type}</span></span>
              <span className="text-text-muted font-mono text-xs">{s.duration_ms.toFixed(0)}ms</span>
              {s.healed && (
                <span className="text-xs font-semibold px-1.5 py-0.5 rounded-md bg-violet-500/15 text-violet-400">healed</span>
              )}
            </div>
          ))}

          {completed && (
            <div className="mt-4 pt-4 border-t border-border text-sm font-semibold flex items-center gap-2">
              Run completed: <StatusBadge status={'status' in completed ? (completed.status ?? null) : 'error'} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default function RunDetail() {
  const { runId } = useParams<{ runId: string }>();
  const [expandedStep, setExpandedStep] = useState<number | null>(null);
  const [showExport, setShowExport] = useState(false);

  const { data: run, isLoading } = useQuery({
    queryKey: ['run', runId],
    queryFn: () => getRun(runId!).then(r => r.data),
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status === 'running' || status === 'pending' ? 3000 : false;
    },
  });

  const isLiveRun = run?.status === 'running' || run?.status === 'pending';

  const { data: steps } = useQuery({
    queryKey: ['run-steps', runId],
    queryFn: () => getRunSteps(runId!).then(r => r.data),
    enabled: !!run && run.status !== 'pending',
    refetchInterval: isLiveRun ? 3000 : false,
  });

  const assertionsByStep = useMemo(
    () => parseAssertions(run?.result_json),
    [run?.result_json],
  );

  const totalAssertions = useMemo(() => {
    let total = 0, passed = 0;
    for (const list of Object.values(assertionsByStep)) {
      total += list.length;
      passed += list.filter(a => a.status === 'passed').length;
    }
    return { total, passed, failed: total - passed };
  }, [assertionsByStep]);

  if (isLoading) return <LoadingSpinner />;
  if (!run) return <div className="text-text-muted text-center py-20">Run not found</div>;

  const isLive = isLiveRun;

  const summaryCards = [
    { label: 'Passed', value: run.passed_count, color: 'text-emerald-400', ring: 'ring-emerald-500/15', bg: 'bg-emerald-500/5' },
    { label: 'Failed', value: run.failed_count, color: 'text-red-400', ring: 'ring-red-500/15', bg: 'bg-red-500/5' },
    { label: 'Healed', value: run.healed_count, color: 'text-violet-400', ring: 'ring-violet-500/15', bg: 'bg-violet-500/5' },
    { label: 'Total Steps', value: run.total_steps, color: 'text-text', ring: 'ring-border', bg: '' },
    {
      label: 'Assertions',
      value: totalAssertions.total > 0 ? `${totalAssertions.passed}/${totalAssertions.total}` : '0',
      color: totalAssertions.failed > 0 ? 'text-red-400' : 'text-text',
      ring: 'ring-border',
      bg: '',
    },
    { label: 'Tokens Used', value: run.tokens_used, color: 'text-text', ring: 'ring-border', bg: '' },
  ];

  return (
    <div className="space-y-6 animate-fade-up">
      {/* Breadcrumb */}
      <Link to="/runs" className="inline-flex items-center gap-1.5 text-sm text-text-muted hover:text-text transition-colors">
        <ArrowLeft className="w-4 h-4" /> Back to Runs
      </Link>

      {/* Header */}
      <div className="flex justify-between items-start gap-4">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="page-title">Run {run.run_id.slice(0, 8)}</h1>
            <StatusBadge status={run.status} />
          </div>
          <div className="flex items-center gap-2 mt-2 text-sm text-text-muted flex-wrap">
            <Link to={`/tests/${run.test_uuid}`} className="hover:text-primary-light transition-colors">{run.test_name}</Link>
            <span className="text-border-light">&middot;</span>
            <span>{run.team_name}</span>
            <span className="text-border-light">&middot;</span>
            <span className="font-mono">v{run.test_version}</span>
            <span className="text-border-light">&middot;</span>
            <span>{run.trigger_type}</span>
            <span className="text-border-light">&middot;</span>
            <span className="font-mono">{run.total_duration_ms > 0 ? `${(run.total_duration_ms / 1000).toFixed(1)}s` : '--'}</span>
          </div>
        </div>

        {!isLive && run.status !== 'error' && (
          <div className="relative">
            <button onClick={() => setShowExport(!showExport)} className="btn btn-ghost">
              <Download className="w-4 h-4" /> Export Report
            </button>
            {showExport && (
              <div className="absolute right-0 mt-2 card py-2 z-10 w-48 animate-scale-in shadow-xl shadow-black/30">
                <a href={getReportUrl(runId!, 'html')} className="flex items-center gap-2.5 px-4 py-2.5 text-sm hover:bg-surface-light transition-colors" download>
                  <FileText className="w-4 h-4 text-text-muted" /> HTML Report
                </a>
                <a href={getReportUrl(runId!, 'json')} className="flex items-center gap-2.5 px-4 py-2.5 text-sm hover:bg-surface-light transition-colors" download>
                  <FileJson className="w-4 h-4 text-text-muted" /> JSON Report
                </a>
                <a href={getReportUrl(runId!, 'junit')} className="flex items-center gap-2.5 px-4 py-2.5 text-sm hover:bg-surface-light transition-colors" download>
                  <FileCode className="w-4 h-4 text-text-muted" /> JUnit XML
                </a>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-3 lg:grid-cols-6 gap-3 stagger-children">
        {summaryCards.map(({ label, value, color, ring, bg }) => (
          <div key={label} className={`card text-center p-4 ${bg} ring-1 ring-inset ${ring}`}>
            <div className={`text-2xl font-bold font-mono ${color}`}>{value}</div>
            <div className="text-[11px] text-text-muted font-semibold uppercase tracking-wider mt-1">{label}</div>
          </div>
        ))}
      </div>

      {/* Live Panel */}
      {isLive && <LivePanel runId={runId!} totalSteps={run.total_steps} />}

      {/* Step Details Table */}
      {steps && steps.length > 0 && (
        <div className="card overflow-hidden">
          <table className="data-table">
            <thead>
              <tr>
                <th>Step</th>
                <th>Action</th>
                <th>Status</th>
                <th>Confidence</th>
                <th>Duration</th>
                <th>Retries</th>
                <th>Assertions</th>
              </tr>
            </thead>
            <tbody>
              {steps.map((step) => (
                <StepRow
                  key={step.step_id}
                  step={step}
                  assertions={assertionsByStep[step.step_id] || []}
                  expanded={expandedStep === step.step_id}
                  onToggle={() => setExpandedStep(expandedStep === step.step_id ? null : step.step_id)}
                />
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Error */}
      {run.error_message && (
        <div className="bg-red-500/5 ring-1 ring-inset ring-red-500/15 rounded-2xl p-5">
          <h3 className="text-sm font-bold text-red-400 mb-1">Error</h3>
          <p className="text-sm text-text-muted">{run.error_message}</p>
        </div>
      )}
    </div>
  );
}
