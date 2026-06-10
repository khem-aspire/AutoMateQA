import { Inbox } from 'lucide-react';

interface EmptyStateProps {
  title: string;
  description?: string;
  action?: React.ReactNode;
}

export function EmptyState({ title, description, action }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center py-20 text-center animate-fade-up">
      <div className="w-16 h-16 rounded-2xl bg-surface-light flex items-center justify-center mb-5">
        <Inbox className="w-7 h-7 text-text-muted" />
      </div>
      <h3 className="text-lg font-display font-semibold text-text mb-1.5">{title}</h3>
      {description && <p className="text-sm text-text-muted mb-5 max-w-sm">{description}</p>}
      {action}
    </div>
  );
}
