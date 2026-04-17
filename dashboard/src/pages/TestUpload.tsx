import { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useQuery, useMutation } from '@tanstack/react-query';
import { uploadTest } from '../api/tests';
import { getTeams, getCategories } from '../api/teams';
import { Upload, FileText, ArrowLeft } from 'lucide-react';
import { Link } from 'react-router-dom';

export default function TestUpload() {
  const navigate = useNavigate();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [ownerEmail, setOwnerEmail] = useState('');
  const [teamId, setTeamId] = useState('');
  const [categoryId, setCategoryId] = useState('');
  const [tags, setTags] = useState('');
  const [error, setError] = useState('');
  const [isDragOver, setIsDragOver] = useState(false);

  const { data: teams } = useQuery({
    queryKey: ['teams'],
    queryFn: () => getTeams().then(r => r.data),
  });

  const { data: categories } = useQuery({
    queryKey: ['categories'],
    queryFn: () => getCategories().then(r => r.data),
  });

  const mutation = useMutation({
    mutationFn: (formData: FormData) => uploadTest(formData),
    onSuccess: (res) => navigate(`/tests/${res.data.test_id}`),
    onError: (err: any) => {
      setError(err.response?.data?.detail || 'Upload failed');
    },
  });

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) {
      setFile(f);
      if (!name) {
        setName(f.name.replace(/\.(aqa|json)$/, ''));
      }
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const f = e.dataTransfer.files?.[0];
    if (f) {
      setFile(f);
      if (!name) {
        setName(f.name.replace(/\.(aqa|json)$/, ''));
      }
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    if (!file) return setError('Please select a test file');
    if (!name.trim()) return setError('Please enter a test name');
    if (!ownerEmail.trim()) return setError('Owner email is required');
    if (!teamId) return setError('Please select a team');

    const formData = new FormData();
    formData.append('file', file);
    formData.append('name', name);
    formData.append('description', description);
    formData.append('owner_email', ownerEmail);
    formData.append('team_id', teamId);
    if (categoryId) formData.append('category_id', categoryId);
    if (tags) formData.append('tags', tags);

    mutation.mutate(formData);
  };

  return (
    <div className="max-w-2xl mx-auto space-y-6 animate-fade-up">
      <div>
        <Link to="/tests" className="inline-flex items-center gap-1.5 text-sm text-text-muted hover:text-text transition-colors mb-4">
          <ArrowLeft className="w-4 h-4" /> Back to Tests
        </Link>
        <h1 className="page-title">Upload Test</h1>
        <p className="text-text-muted text-sm mt-1">Add a new test file to your suite</p>
      </div>

      <form onSubmit={handleSubmit} className="card p-7 space-y-6">
        {/* File Drop Zone */}
        <div
          onClick={() => fileInputRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
          onDragLeave={() => setIsDragOver(false)}
          onDrop={handleDrop}
          className={`border-2 border-dashed rounded-2xl p-10 text-center cursor-pointer transition-all duration-200 ${
            isDragOver
              ? 'border-primary bg-primary/5'
              : file
                ? 'border-primary/30 bg-primary/5'
                : 'border-border hover:border-border-light hover:bg-surface-light/30'
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".aqa,.json"
            onChange={handleFileChange}
            className="hidden"
          />
          {file ? (
            <div className="flex items-center justify-center gap-4">
              <div className="w-14 h-14 rounded-2xl bg-primary/10 flex items-center justify-center">
                <FileText className="w-7 h-7 text-primary" />
              </div>
              <div className="text-left">
                <p className="font-semibold text-text">{file.name}</p>
                <p className="text-sm text-text-muted mt-0.5">{(file.size / 1024).toFixed(1)} KB</p>
              </div>
            </div>
          ) : (
            <div>
              <div className="w-14 h-14 rounded-2xl bg-surface-light flex items-center justify-center mx-auto mb-3">
                <Upload className="w-7 h-7 text-text-muted" />
              </div>
              <p className="text-sm font-medium text-text">Drop your test file here</p>
              <p className="text-xs text-text-muted mt-1">Supports .aqa and .json files</p>
            </div>
          )}
        </div>

        {/* Form Fields */}
        <div className="grid grid-cols-1 gap-5">
          <div>
            <label className="block text-sm font-semibold text-text mb-2">Test Name <span className="text-danger">*</span></label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="input-field"
              placeholder="e.g. Login Flow"
            />
          </div>

          <div>
            <label className="block text-sm font-semibold text-text mb-2">Description</label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              rows={3}
              className="input-field resize-none"
              placeholder="Describe what this test covers..."
            />
          </div>

          <div>
            <label className="block text-sm font-semibold text-text mb-2">Owner Email <span className="text-danger">*</span></label>
            <input
              type="email"
              value={ownerEmail}
              onChange={(e) => setOwnerEmail(e.target.value)}
              className="input-field"
              placeholder="owner@company.com"
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-semibold text-text mb-2">Team <span className="text-danger">*</span></label>
              <select
                value={teamId}
                onChange={(e) => setTeamId(e.target.value)}
                className="input-field"
              >
                <option value="">Select team...</option>
                {teams?.map(t => <option key={t.id} value={t.id}>{t.name}</option>)}
              </select>
            </div>
            <div>
              <label className="block text-sm font-semibold text-text mb-2">Category</label>
              <select
                value={categoryId}
                onChange={(e) => setCategoryId(e.target.value)}
                className="input-field"
              >
                <option value="">Select category...</option>
                {categories?.map(c => <option key={c.id} value={c.id}>{c.name}</option>)}
              </select>
            </div>
          </div>

          <div>
            <label className="block text-sm font-semibold text-text mb-2">Tags</label>
            <input
              type="text"
              value={tags}
              onChange={(e) => setTags(e.target.value)}
              className="input-field"
              placeholder="Comma-separated tags"
            />
          </div>
        </div>

        {error && (
          <div className="px-4 py-3 bg-danger/10 border border-danger/20 rounded-xl text-sm text-danger font-medium">
            {error}
          </div>
        )}

        <button
          type="submit"
          disabled={mutation.isPending}
          className="btn btn-primary w-full justify-center py-3 text-base"
        >
          {mutation.isPending ? (
            <>
              <div className="w-4 h-4 rounded-full border-2 border-white/30 border-t-white animate-spin" />
              Uploading...
            </>
          ) : 'Upload Test'}
        </button>
      </form>
    </div>
  );
}
