import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Layout } from './components/layout/Layout';
import Dashboard from './pages/Dashboard';
import Tests from './pages/Tests';
import TestUpload from './pages/TestUpload';
import TestDetail from './pages/TestDetail';
import Runs from './pages/Runs';
import RunDetail from './pages/RunDetail';
import Schedules from './pages/Schedules';
import Categories from './pages/Categories';
import Teams from './pages/Teams';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30_000,
      retry: 1,
    },
  },
});

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route element={<Layout />}>
            <Route path="/" element={<Dashboard />} />
            <Route path="/tests" element={<Tests />} />
            <Route path="/tests/upload" element={<TestUpload />} />
            <Route path="/tests/:testUuid" element={<TestDetail />} />
            <Route path="/runs" element={<Runs />} />
            <Route path="/runs/:runId" element={<RunDetail />} />
            <Route path="/schedules" element={<Schedules />} />
            <Route path="/categories" element={<Categories />} />
            <Route path="/teams" element={<Teams />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
