const statusConfig: Record<string, { bg: string; text: string; ring: string; label: string; dot?: string }> = {
  passed: {
    bg: 'bg-emerald-500/10',
    text: 'text-emerald-400',
    ring: 'ring-emerald-500/20',
    label: 'Passed',
    dot: 'bg-emerald-400',
  },
  failed: {
    bg: 'bg-red-500/10',
    text: 'text-red-400',
    ring: 'ring-red-500/20',
    label: 'Failed',
    dot: 'bg-red-400',
  },
  healed: {
    bg: 'bg-violet-500/10',
    text: 'text-violet-400',
    ring: 'ring-violet-500/20',
    label: 'Healed',
    dot: 'bg-violet-400',
  },
  running: {
    bg: 'bg-sky-500/10',
    text: 'text-sky-400',
    ring: 'ring-sky-500/20',
    label: 'Running',
  },
  pending: {
    bg: 'bg-zinc-500/10',
    text: 'text-zinc-400',
    ring: 'ring-zinc-500/20',
    label: 'Pending',
    dot: 'bg-zinc-400',
  },
  error: {
    bg: 'bg-red-500/10',
    text: 'text-red-400',
    ring: 'ring-red-500/20',
    label: 'Error',
    dot: 'bg-red-400',
  },
};

export function StatusBadge({ status }: { status: string | null }) {
  const config = statusConfig[status || ''] || statusConfig.pending;
  return (
    <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-semibold ring-1 ring-inset ${config.bg} ${config.text} ${config.ring}`}>
      {status === 'running' ? (
        <span className="relative flex h-2 w-2">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-sky-400 opacity-75" />
          <span className="relative inline-flex rounded-full h-2 w-2 bg-sky-400" />
        </span>
      ) : config.dot ? (
        <span className={`w-1.5 h-1.5 rounded-full ${config.dot}`} />
      ) : null}
      {config.label}
    </span>
  );
}
