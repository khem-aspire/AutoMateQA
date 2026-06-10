import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { getCategories, createCategory, updateCategory, deleteCategory } from '../api/teams';
import { LoadingSpinner } from '../components/shared/LoadingSpinner';
import { Plus, Pencil, Trash2, Tag } from 'lucide-react';

export default function Categories() {
  const queryClient = useQueryClient();
  const [editing, setEditing] = useState<number | null>(null);
  const [newName, setNewName] = useState('');
  const [newDesc, setNewDesc] = useState('');
  const [newColor, setNewColor] = useState('#10B981');
  const [showAdd, setShowAdd] = useState(false);

  const { data: categories, isLoading } = useQuery({
    queryKey: ['categories'],
    queryFn: () => getCategories().then(r => r.data),
  });

  const createMut = useMutation({
    mutationFn: () => createCategory({ name: newName, description: newDesc, color: newColor }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['categories'] });
      setShowAdd(false);
      setNewName('');
      setNewDesc('');
    },
  });

  const updateMut = useMutation({
    mutationFn: ({ id, data }: { id: number; data: Record<string, unknown> }) => updateCategory(id, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['categories'] });
      setEditing(null);
    },
  });

  const deleteMut = useMutation({
    mutationFn: (id: number) => deleteCategory(id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['categories'] }),
  });

  if (isLoading) return <LoadingSpinner />;

  return (
    <div className="space-y-6 animate-fade-up">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="page-title">Categories</h1>
          <p className="text-text-muted text-sm mt-1">Organize tests by category</p>
        </div>
        <button onClick={() => setShowAdd(!showAdd)} className="btn btn-primary">
          <Plus className="w-4 h-4" /> Add Category
        </button>
      </div>

      {showAdd && (
        <div className="card p-5 animate-scale-in">
          <div className="flex gap-3 items-end">
            <div className="flex-1">
              <label className="block text-xs font-semibold text-text-muted uppercase tracking-wider mb-1.5">Name</label>
              <input
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                className="input-field"
                placeholder="e.g. Smoke Tests"
              />
            </div>
            <div className="flex-1">
              <label className="block text-xs font-semibold text-text-muted uppercase tracking-wider mb-1.5">Description</label>
              <input
                value={newDesc}
                onChange={(e) => setNewDesc(e.target.value)}
                className="input-field"
                placeholder="Optional description"
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
        {categories?.map((cat) => (
          <div key={cat.id} className="card-glow p-5 group">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-3">
                <div
                  className="w-10 h-10 rounded-xl flex items-center justify-center"
                  style={{ backgroundColor: cat.color + '18' }}
                >
                  <Tag className="w-4.5 h-4.5" style={{ color: cat.color }} />
                </div>
                <h3 className="font-semibold">{cat.name}</h3>
              </div>
              <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                <button onClick={() => setEditing(cat.id)} className="text-text-muted hover:text-text p-1.5 rounded-lg hover:bg-surface-light transition-colors">
                  <Pencil className="w-3.5 h-3.5" />
                </button>
                <button
                  onClick={() => { if (window.confirm('Delete this category?')) deleteMut.mutate(cat.id); }}
                  className="text-text-muted hover:text-red-400 p-1.5 rounded-lg hover:bg-red-500/10 transition-colors"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
            <p className="text-sm text-text-muted mb-3">{cat.description || 'No description'}</p>
            <div className="flex items-center gap-2">
              <span className="text-xs font-semibold px-2.5 py-1 rounded-lg bg-surface-light text-text-muted">{cat.test_count} tests</span>
            </div>

            {editing === cat.id && (
              <div className="mt-4 pt-4 border-t border-border space-y-2 animate-fade-in">
                <input
                  defaultValue={cat.name}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      updateMut.mutate({ id: cat.id, data: { name: e.currentTarget.value } });
                    }
                  }}
                  className="input-field"
                  autoFocus
                />
                <button onClick={() => setEditing(null)} className="text-xs text-text-muted hover:text-text transition-colors">
                  Cancel (press Enter to save)
                </button>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
