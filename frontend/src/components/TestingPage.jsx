import { useState, useMemo } from "react";

const API = "http://localhost:8000";
const PAGE_SIZE = 50;

const NON_PCT = new Set(["mae", "mse", "rmse", "r2_score", "silhouette_score", "error"]);

function MetricCard({ label, value }) {
  if (typeof value !== "number") return null;
  const isPct = value >= 0 && value <= 1 && !NON_PCT.has(label.toLowerCase());
  const formatted = isPct ? (value * 100).toFixed(1) + "%" : value.toFixed(4);
  const highlight = ["accuracy", "f1", "r2_score"].includes(label.toLowerCase());
  return (
    <div className={`rounded-xl p-4 border ${highlight ? "border-brand-500 bg-brand-900/20" : "border-gray-700 bg-gray-800"}`}>
      <div className="text-xs text-gray-400 mb-1 uppercase tracking-wide">{label}</div>
      <div className={`text-2xl font-bold ${highlight ? "text-brand-400" : "text-white"}`}>{formatted}</div>
    </div>
  );
}

function StatusBadge({ row, taskType }) {
  if (taskType?.includes("classif")) {
    if (row.correct === undefined || row.correct === null) return <span className="text-gray-500">—</span>;
    return row.correct
      ? <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-semibold bg-green-900/50 text-green-400 border border-green-700">✓ Correct</span>
      : <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-semibold bg-red-900/50 text-red-400 border border-red-700">✗ Wrong</span>;
  }
  if (taskType?.includes("regress")) {
    const err = row.error;
    if (err === undefined || err === null) return <span className="text-gray-500">—</span>;
    return <span className="text-gray-300 font-mono text-sm">{err > 0 ? "+" : ""}{Number(err).toFixed(4)}</span>;
  }
  return <span className="inline-flex px-2 py-0.5 rounded-full text-xs font-semibold bg-blue-900/50 text-blue-300 border border-blue-700">Cluster {row.cluster ?? row.predicted}</span>;
}

export default function TestingPage({ session, onBack }) {
  const [file, setFile]       = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError]     = useState("");
  const [page, setPage]       = useState(0);
  const [search, setSearch]   = useState("");

  const taskType = session?.taskType || "";
  const isClassif  = taskType.includes("classif");
  const isRegress  = taskType.includes("regress");
  const isCluster  = !isClassif && !isRegress;

  const handleFile = (e) => {
    const f = e.target.files?.[0];
    if (f) { setFile(f); setResults(null); setError(""); }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const f = e.dataTransfer.files?.[0];
    if (f?.name.endsWith(".csv")) { setFile(f); setResults(null); setError(""); }
  };

  const handleTest = async () => {
    if (!file) return;
    setLoading(true); setError(""); setResults(null); setPage(0);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const res = await fetch(`${API}/test/${session.sessionId}`, { method: "POST", body: fd });
      if (!res.ok) {
        const j = await res.json().catch(() => ({}));
        throw new Error(j.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      setResults(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  // Derive column list from first prediction row
  const featureCols = useMemo(() => {
    if (!results?.predictions?.length) return [];
    const skip = new Set(["actual", "predicted", "correct", "error", "cluster"]);
    return Object.keys(results.predictions[0]).filter(k => !skip.has(k));
  }, [results]);

  const filteredRows = useMemo(() => {
    if (!results?.predictions) return [];
    if (!search.trim()) return results.predictions;
    const q = search.toLowerCase();
    return results.predictions.filter(row =>
      Object.values(row).some(v => String(v).toLowerCase().includes(q))
    );
  }, [results, search]);

  const pageRows  = filteredRows.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);
  const totalPages = Math.ceil(filteredRows.length / PAGE_SIZE);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-4">
        <button onClick={onBack} className="text-gray-400 hover:text-white transition-colors text-sm flex items-center gap-1">
          ← Back to Results
        </button>
        <h2 className="text-xl font-bold text-white">Test Model on New Data</h2>
      </div>

      {/* Upload Zone */}
      <div
        onDrop={handleDrop}
        onDragOver={e => e.preventDefault()}
        className="border-2 border-dashed border-gray-600 rounded-xl p-8 text-center hover:border-brand-500 transition-colors"
      >
        <div className="text-4xl mb-3">📂</div>
        <p className="text-gray-300 mb-2">Drag & drop a test CSV, or click to browse</p>
        <p className="text-gray-500 text-sm mb-4">Must have the same feature columns as the training data</p>
        <label className="cursor-pointer">
          <input type="file" accept=".csv" onChange={handleFile} className="hidden" />
          <span className="bg-gray-700 hover:bg-gray-600 text-white px-4 py-2 rounded-lg text-sm transition-colors">
            Browse File
          </span>
        </label>
        {file && (
          <div className="mt-3 text-brand-400 text-sm font-medium">
            Selected: {file.name} ({(file.size / 1024).toFixed(1)} KB)
          </div>
        )}
      </div>

      {/* Run Button */}
      <button
        onClick={handleTest}
        disabled={!file || loading}
        className="w-full bg-brand-600 hover:bg-brand-700 disabled:opacity-50 disabled:cursor-not-allowed text-white font-semibold py-3 rounded-lg transition-colors flex items-center justify-center gap-2"
      >
        {loading ? (
          <><span className="animate-spin">⚙️</span> Running predictions…</>
        ) : (
          <>▶ Run Test Predictions</>
        )}
      </button>

      {error && (
        <div className="bg-red-950 border border-red-700 text-red-300 px-4 py-3 rounded-lg text-sm">
          {error}
        </div>
      )}

      {/* Results */}
      {results && (
        <div className="space-y-5">
          {/* Summary bar */}
          <div className="bg-gray-800 border border-gray-700 rounded-xl p-4 flex flex-wrap gap-4 items-center">
            <div>
              <span className="text-gray-400 text-sm">Total rows: </span>
              <span className="text-white font-bold">{results.summary?.total ?? results.predictions?.length}</span>
            </div>
            {isClassif && results.summary?.correct !== undefined && (
              <>
                <div>
                  <span className="text-green-400 font-bold">{results.summary.correct}</span>
                  <span className="text-gray-400 text-sm"> correct</span>
                </div>
                <div>
                  <span className="text-red-400 font-bold">{results.summary.wrong}</span>
                  <span className="text-gray-400 text-sm"> wrong</span>
                </div>
                <div>
                  <span className="text-brand-400 font-bold">{results.summary.accuracy_pct}%</span>
                  <span className="text-gray-400 text-sm"> accuracy</span>
                </div>
              </>
            )}
            {isCluster && results.cluster_counts && (
              <div className="flex gap-2 flex-wrap">
                {Object.entries(results.cluster_counts).map(([k, v]) => (
                  <span key={k} className="px-2 py-0.5 rounded-full bg-blue-900/50 text-blue-300 text-xs border border-blue-700">
                    Cluster {k}: {v} rows
                  </span>
                ))}
              </div>
            )}
            {!results.has_actual_labels && (
              <span className="text-yellow-400 text-sm">⚠ Test CSV has no target column — metrics unavailable</span>
            )}
          </div>

          {/* Metric cards */}
          {results.metrics && Object.keys(results.metrics).length > 0 && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {Object.entries(results.metrics)
                .filter(([, v]) => typeof v === "number")
                .map(([k, v]) => <MetricCard key={k} label={k} value={v} />)}
            </div>
          )}

          {/* Classification report */}
          {results.metrics?.classification_report && (
            <div className="bg-gray-900 border border-gray-700 rounded-xl p-4">
              <div className="text-gray-400 text-xs font-semibold mb-2 uppercase tracking-wide">Classification Report</div>
              <pre className="text-gray-300 text-xs font-mono overflow-x-auto whitespace-pre">
                {results.metrics.classification_report}
              </pre>
            </div>
          )}

          {/* Search + Pagination header */}
          <div className="flex items-center justify-between gap-4 flex-wrap">
            <input
              type="text"
              placeholder="Search predictions…"
              value={search}
              onChange={e => { setSearch(e.target.value); setPage(0); }}
              className="bg-gray-800 border border-gray-600 rounded-lg px-3 py-2 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-brand-500 w-64"
            />
            <div className="text-gray-400 text-sm">
              Showing {filteredRows.length} row{filteredRows.length !== 1 ? "s" : ""}
              {search && " (filtered)"}
            </div>
          </div>

          {/* Predictions table */}
          <div className="overflow-x-auto rounded-xl border border-gray-700">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-gray-800 border-b border-gray-700">
                  <th className="px-3 py-2 text-left text-gray-400 font-medium">#</th>
                  {featureCols.map(col => (
                    <th key={col} className="px-3 py-2 text-left text-gray-400 font-medium max-w-[120px] truncate">{col}</th>
                  ))}
                  {!isCluster && results.has_actual_labels && (
                    <th className="px-3 py-2 text-left text-gray-400 font-medium">Actual</th>
                  )}
                  <th className="px-3 py-2 text-left text-gray-400 font-medium">Predicted</th>
                  <th className="px-3 py-2 text-left text-gray-400 font-medium">
                    {isClassif ? "Status" : isRegress ? "Error" : "Cluster"}
                  </th>
                </tr>
              </thead>
              <tbody>
                {pageRows.map((row, idx) => {
                  const absIdx = page * PAGE_SIZE + idx;
                  const rowBg = isClassif
                    ? (row.correct === false ? "bg-red-950/20 hover:bg-red-950/30" : row.correct === true ? "bg-green-950/20 hover:bg-green-950/30" : "hover:bg-gray-800")
                    : "hover:bg-gray-800";
                  return (
                    <tr key={absIdx} className={`border-b border-gray-800 transition-colors ${rowBg}`}>
                      <td className="px-3 py-2 text-gray-500 font-mono text-xs">{absIdx}</td>
                      {featureCols.map(col => (
                        <td key={col} className="px-3 py-2 text-gray-300 font-mono text-xs max-w-[120px] truncate">
                          {row[col] !== undefined ? String(row[col]) : "—"}
                        </td>
                      ))}
                      {!isCluster && results.has_actual_labels && (
                        <td className="px-3 py-2 text-gray-200 font-mono text-xs">{row.actual ?? "—"}</td>
                      )}
                      <td className="px-3 py-2 text-white font-mono text-xs font-semibold">
                        {row.predicted ?? row.cluster ?? "—"}
                      </td>
                      <td className="px-3 py-2">
                        <StatusBadge row={row} taskType={taskType} />
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="flex items-center justify-center gap-2">
              <button
                onClick={() => setPage(p => Math.max(0, p - 1))}
                disabled={page === 0}
                className="px-3 py-1.5 rounded-lg bg-gray-800 border border-gray-600 text-sm disabled:opacity-40 hover:border-brand-500 transition-colors"
              >
                ←
              </button>
              <span className="text-gray-400 text-sm">Page {page + 1} / {totalPages}</span>
              <button
                onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))}
                disabled={page === totalPages - 1}
                className="px-3 py-1.5 rounded-lg bg-gray-800 border border-gray-600 text-sm disabled:opacity-40 hover:border-brand-500 transition-colors"
              >
                →
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
