import { useEffect, useState } from "react";
import UploadScreen from "./components/UploadScreen";
import PipelineDashboard from "./components/PipelineDashboard";

const SESSION_KEY = "ada_session";
const API = "http://localhost:8000";

export default function App() {
  const [session, setSession]   = useState(null);
  const [restoring, setRestoring] = useState(true);

  // On mount, attempt to restore a saved session from localStorage
  useEffect(() => {
    const saved = localStorage.getItem(SESSION_KEY);
    if (!saved) { setRestoring(false); return; }

    let parsed;
    try { parsed = JSON.parse(saved); } catch { localStorage.removeItem(SESSION_KEY); setRestoring(false); return; }

    // Validate the session still exists on the backend before restoring
    fetch(`${API}/state/${parsed.sessionId}`)
      .then((r) => r.ok ? r.json() : Promise.reject())
      .then(() => { setSession(parsed); })
      .catch(() => { localStorage.removeItem(SESSION_KEY); })
      .finally(() => setRestoring(false));
  }, []);

  const handleStart = (s) => {
    localStorage.setItem(SESSION_KEY, JSON.stringify(s));
    setSession(s);
  };

  const handleReset = () => {
    localStorage.removeItem(SESSION_KEY);
    setSession(null);
  };

  if (restoring) {
    return (
      <div className="min-h-screen bg-gray-950 flex items-center justify-center">
        <div className="text-gray-500 text-sm animate-pulse">Restoring session…</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      <header className="border-b border-gray-800 px-6 py-4 flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg bg-brand-600 flex items-center justify-center font-bold text-white text-sm">
          AI
        </div>
        <span className="font-semibold text-white text-lg">Ada: 24/7 Available Data Scientist</span>
      </header>

      <main className="max-w-6xl mx-auto px-4 py-8">
        {!session ? (
          <UploadScreen onStart={handleStart} />
        ) : (
          <PipelineDashboard session={session} onReset={handleReset} />
        )}
      </main>
    </div>
  );
}
