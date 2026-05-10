import { useEffect, useState } from "react";
import AnalysisViewer from "./AnalysisViewer";

const API = "http://localhost:8000";

function PlotImage({ sessionId, filename }) {
  const [status, setStatus] = useState("loading"); // loading | ok | error

  return (
    <div className="bg-gray-900 border border-gray-700 rounded-xl overflow-hidden">
      <div className="px-4 py-2 border-b border-gray-700 flex items-center justify-between">
        <span className="text-xs font-mono text-gray-400">{filename}</span>
        <a
          href={`${API}/outputs/${sessionId}/plot/${filename}`}
          target="_blank"
          rel="noreferrer"
          className="text-xs text-brand-400 hover:text-brand-300 transition-colors"
        >
          open full size ↗
        </a>
      </div>
      <div className="relative min-h-[200px] flex items-center justify-center bg-gray-950">
        {status === "loading" && (
          <span className="text-gray-600 text-sm animate-pulse">Loading…</span>
        )}
        {status === "error" && (
          <span className="text-red-500 text-sm">Failed to load image</span>
        )}
        <img
          src={`${API}/outputs/${sessionId}/plot/${filename}`}
          alt={filename}
          onLoad={() => setStatus("ok")}
          onError={() => setStatus("error")}
          className={`w-full object-contain max-h-[480px] transition-opacity duration-300 ${
            status === "ok" ? "opacity-100" : "opacity-0 absolute"
          }`}
        />
      </div>
    </div>
  );
}

export default function AnalysisTab({ sessionId, analysisData, ready }) {
  const [plots, setPlots] = useState([]);
  const [fetchingPlots, setFetchingPlots] = useState(false);

  const loadPlots = async () => {
    setFetchingPlots(true);
    try {
      const res = await fetch(`${API}/outputs/${sessionId}/plots`);
      if (res.ok) {
        const { plots: list } = await res.json();
        setPlots(list || []);
      }
    } catch (_) {}
    setFetchingPlots(false);
  };

  // Fetch plot list once analysis is ready and tab mounts
  useEffect(() => {
    if (!ready) return;
    loadPlots();
  }, [ready, sessionId]);

  if (!ready) {
    return (
      <div className="text-gray-500 text-sm italic text-center py-16">
        Analysis results will appear here after the Data Analysis step completes.
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* ── Plots ─────────────────────────────────────────────── */}
      {(plots.length > 0 || ready) && (
        <section className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wider">
              Generated Plots {plots.length > 0 && `(${plots.length})`}
            </h3>
            <button
              onClick={loadPlots}
              disabled={fetchingPlots}
              className="text-xs text-gray-400 hover:text-white border border-gray-700 hover:border-gray-500 px-3 py-1 rounded-lg transition-colors disabled:opacity-40"
            >
              {fetchingPlots ? "Refreshing…" : "↺ Refresh"}
            </button>
          </div>
          {plots.length > 0 ? (
            <div className="grid md:grid-cols-2 gap-6">
              {plots.map((f) => (
                <PlotImage key={f} sessionId={sessionId} filename={f} />
              ))}
            </div>
          ) : (
            <div className="text-gray-600 text-sm italic text-center py-6 border border-dashed border-gray-700 rounded-xl">
              No plots found — click Refresh if the analysis just completed.
            </div>
          )}
        </section>
      )}

      {/* ── Structured analysis output ────────────────────────── */}
      {analysisData && Object.keys(analysisData).length > 0 && (
        <section className="space-y-4">
          <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wider">
            Analysis Summary
          </h3>
          <div className="bg-gray-900 border border-gray-700 rounded-xl p-5">
            <AnalysisViewer data={analysisData} />
          </div>
        </section>
      )}

      {/* Empty state */}
      {plots.length === 0 && (!analysisData || Object.keys(analysisData).length === 0) && (
        <div className="text-gray-500 text-sm italic text-center py-16">
          No analysis output found.
        </div>
      )}
    </div>
  );
}
