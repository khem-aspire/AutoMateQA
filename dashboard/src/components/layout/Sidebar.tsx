import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  TestTube2,
  Play,
  Clock,
  Tag,
  Users,
  Hexagon,
} from 'lucide-react';

const navItems = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/tests', icon: TestTube2, label: 'Tests' },
  { to: '/runs', icon: Play, label: 'Runs' },
  { to: '/schedules', icon: Clock, label: 'Schedules' },
  { to: '/categories', icon: Tag, label: 'Categories' },
  { to: '/teams', icon: Users, label: 'Teams' },
];

export function Sidebar() {
  return (
    <aside className="w-[260px] bg-surface border-r border-border flex flex-col shrink-0 relative overflow-hidden">
      {/* Subtle gradient overlay */}
      <div className="absolute inset-0 bg-gradient-to-b from-primary-glow/40 via-transparent to-transparent pointer-events-none" />

      {/* Brand */}
      <div className="relative z-10 px-6 py-7 border-b border-border">
        <div className="flex items-center gap-3">
          <div className="relative">
            <div className="w-10 h-10 rounded-xl bg-primary/15 flex items-center justify-center animate-pulse-glow">
              <Hexagon className="w-5 h-5 text-primary" strokeWidth={2.5} />
            </div>
          </div>
          <div>
            <h1 className="font-display text-lg font-bold tracking-tight text-text">
              AutoMateQA
            </h1>
            <p className="text-[11px] font-medium text-text-muted tracking-wide uppercase">
              Self-Healing Engine
            </p>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="relative z-10 flex-1 px-4 py-5 space-y-1">
        {navItems.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              `group flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all duration-200 relative ${
                isActive
                  ? 'bg-primary/10 text-primary-light'
                  : 'text-text-muted hover:text-text hover:bg-surface-light'
              }`
            }
          >
            {({ isActive }) => (
              <>
                {/* Active indicator bar */}
                {isActive && (
                  <div className="absolute left-0 top-1/2 -translate-y-1/2 w-[3px] h-5 rounded-r-full bg-primary animate-slide-in" />
                )}
                <Icon className={`w-[18px] h-[18px] transition-colors ${isActive ? 'text-primary' : 'text-text-muted group-hover:text-text-secondary'}`} />
                <span className="tracking-wide">{label}</span>
              </>
            )}
          </NavLink>
        ))}
      </nav>

      {/* Footer */}
      <div className="relative z-10 px-6 py-5 border-t border-border">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-primary animate-pulse" />
          <span className="text-xs text-text-muted font-medium">System Operational</span>
        </div>
      </div>
    </aside>
  );
}
