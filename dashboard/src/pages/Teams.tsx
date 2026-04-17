import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { getTeams, createTeam, deleteTeam } from '../api/teams';
import { LoadingSpinner } from '../components/shared/LoadingSpinner';
import { Plus, Trash2, Users } from 'lucide-react';

export default function Teams() {
  const queryClient = useQueryClient();
  const [showAdd, setShowAdd] = useState(false);
  const [newName, setNewName] = useState('');
  const [newSlug, setNewSlug] = useState('');
  const [newDesc, setNewDesc] = useState('');
  const [newColor, setNewColor] = useState('#10B981');

  const { data: teams, isLoading } = useQuery({
    queryKey: ['teams'],
    queryFn: () => getTeams().then(r => r.data),
  });

  const createMut = useMutation({
    mutationFn: () => createTeam({ name: newName, slug: newSlug || newName.toLowerCase().replace(/\s+/g, '-'), description: newDesc, color: newColor }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['teams'] });
      setShowAdd(false);
      setNewName('');
      setNewSlug('');
      setNewDesc('');
    },
  });

  const deleteMut = useMutation({
    mutationFn: (id: number) => deleteTeam(id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['teams'] }),
    onError: (err: any) => alert(err.response?.data?.detail || 'Cannot delete team'),
  });

  if (isLoading) return <LoadingSpinner />;

  return (
    <div className="space-y-6 animate-fade-up">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="page-title">Teams</h1>
          <p className="text-text-muted text-sm mt-1">Manage your testing teams</p>
        </div>
        <button onClick={() => setShowAdd(!showAdd)} className="btn btn-primary">
          <Plus className="w-4 h-4" /> Add Team
        </button>
      </div>

      {showAdd && (
        <div className="card p-5 space-y-4 animate-scale-in">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-semibold text-text-muted uppercase tracking-wider mb-1.5">Name</label>
              <input
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                className="input-field"
                placeholder="e.g. AspireOS"
              />
            </div>
            <div>
              <label className="block text-xs font-semibold text-text-muted uppercase tracking-wider mb-1.5">Slug</label>
              <input
                value={newSlug}
                onChange={(e) => setNewSlug(e.target.value)}
                className="input-field input-mono"
                placeholder="e.g. aspireos (auto-generated)"
              />
            </div>
          </div>
          <div className="flex gap-4 items-end">
            <div className="flex-1">
              <label className="block text-xs font-semibold text-text-muted uppercase tracking-wider mb-1.5">Description</label>
              <input
                value={newDesc}
                onChange={(e) => setNewDesc(e.target.value)}
                className="input-field"
                placeholder="What does this team own?"
              />
            </div>
            <div>
              <label className="block text-xs font-semibold text-text-muted uppercase tracking-wider mb-1.5">Color</label>
              <input type="color" value={newColor} onChange={(e) => setNewColor(e.target.value)} className="h-[42px] w-14 rounded-xl cursor-pointer border border-border bg-bg" />
            </div>
            <button
              onClick={() => createMut.mutate()}
              disabled={!newName.trim()}
              className="btn btn-success"
            >
              Save
            </button>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 stagger-children">
        {teams?.map((team) => (
          <div key={team.id} className="card-glow p-5 group">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <div
                  className="w-11 h-11 rounded-xl flex items-center justify-center transition-shadow duration-300 group-hover:shadow-lg"
                  style={{
                    backgroundColor: team.color + '18',
                    boxShadow: undefined,
                  }}
                >
                  <Users className="w-5 h-5" style={{ color: team.color }} />
                </div>
                <div>
                  <h3 className="font-semibold">{team.name}</h3>
                  <p className="text-xs text-text-muted font-mono">@{team.slug}</p>
                </div>
              </div>
              <button
                onClick={() => {
                  if (window.confirm(`Delete team "${team.name}"? Tests must be reassigned first.`)) deleteMut.mutate(team.id);
                }}
                className="text-text-muted hover:text-red-400 p-1.5 rounded-lg hover:bg-red-500/10 transition-colors opacity-0 group-hover:opacity-100"
              >
                <Trash2 className="w-4 h-4" />
              </button>
            </div>
            <p className="text-sm text-text-muted mb-4">{team.description || 'No description'}</p>
            <div className="flex items-center gap-3">
              <span className="text-xs font-semibold px-2.5 py-1 rounded-lg bg-surface-light text-text-muted">{team.test_count} tests</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
