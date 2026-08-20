import { Outlet, Route, Routes } from 'react-router-dom';
import Layout from './components/Layout';
import Landing from './pages/Landing';
import RunsDashboard from './pages/RunsDashboard';
import SubmitJob from './pages/SubmitJob';
import RunDetail from './pages/RunDetail';
import Deployments from './pages/Deployments';

/** The marketing landing lives at "/" with its own chrome. Everything else is
 * the product app, wrapped in the shared Layout (header + nav + tour). */
function AppChrome() {
  return (
    <Layout>
      <Outlet />
    </Layout>
  );
}

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Landing />} />
      <Route element={<AppChrome />}>
        <Route path="/dashboard" element={<RunsDashboard />} />
        <Route path="/submit" element={<SubmitJob />} />
        <Route path="/runs/:runId" element={<RunDetail />} />
        <Route path="/deployments" element={<Deployments />} />
      </Route>
    </Routes>
  );
}
