import { Route, Routes } from 'react-router-dom';
import Layout from './components/Layout';
import RunsDashboard from './pages/RunsDashboard';
import SubmitJob from './pages/SubmitJob';
import RunDetail from './pages/RunDetail';
import Deployments from './pages/Deployments';

// All four routes are wired here by the foundation. Page agents implement the
// page components (imported above) but must NOT edit this file.
export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<RunsDashboard />} />
        <Route path="/submit" element={<SubmitJob />} />
        <Route path="/runs/:runId" element={<RunDetail />} />
        <Route path="/deployments" element={<Deployments />} />
      </Routes>
    </Layout>
  );
}
