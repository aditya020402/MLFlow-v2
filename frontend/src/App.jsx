import { useState } from "react";
import UploadScreen from "./components/UploadScreen";
import PipelineDashboard from "./components/PipelineDashboard";

export default function App() {
  const [session, setSession] = useState(null); // { sessionId, taskType, targetColumn }

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
          <UploadScreen onStart={(s) => setSession(s)} />
        ) : (
          <PipelineDashboard session={session} onReset={() => setSession(null)} />
        )}
      </main>
    </div>
  );
}
