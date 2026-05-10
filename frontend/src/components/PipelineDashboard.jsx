import { useEffect, useRef, useState } from "react";
import AgentConsole from "./AgentConsole";
import AnalysisTab from "./AnalysisTab";
import AnalysisViewer from "./AnalysisViewer";
import CodeViewer from "./CodeViewer";
import UnderstandingTab from "./UnderstandingTab";

const NON_PCT = new Set(["mae","mse","rmse","mean_absolute_error","mean_squared_error","root_mean_squared_error","r2_score","silhouette_score"]);
const fmtMetric = (k, v) => {
  if (typeof v !== "number") return String(v ?? "—");
  return (v >= 0 && v <= 1 && !NON_PCT.has(k.toLowerCase())) ? (v * 100).toFixed(1) + "%" : v.toFixed(4);
};

function ValidationTab({ results }) {
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(0);
  const PAGE_SIZE = 50;

  if (!results) {
    return (
      <div className="text-gray-500 text-sm italic text-center py-12">
        Validation results will appear here after the pipeline completes.
      </div>
    );
  }

  const { task_type, predictions = [], metrics = {}, summary = {}, cluster_counts } = results;
  const isClassif   = task_type?.includes("classif");
  const isRegress   = task_type?.includes("regress");
  const isClustering = task_type?.includes("cluster");

  const RESULT_KEYS = new Set(["actual", "predicted", "correct", "error", "cluster"]);
  const featureCols = predictions.length > 0
    ? Object.keys(predictions[0]).filter(k => !RESULT_KEYS.has(k))
    : [];

  const filtered = search
    ? predictions.filter(row =>
        Object.values(row).some(v => String(v).toLowerCase().includes(search.toLowerCase()))
      )
    : predictions;

  const totalPages = Math.ceil(filtered.length / PAGE_SIZE);
  const pageRows   = filtered.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);
  const numericMetrics = Object.entries(metrics).filter(([, v]) => typeof v === "number");

  return (
    <div className="space-y-5">
      {/* Summary */}
      <div className="bg-purple-950/30 border border-purple-700 rounded-xl p-5 space-y-3">
        <div className="text-purple-300 font-semibold text-sm">5% Holdout Validation Summary</div>
        {isClassif && (
          <div className="flex flex-wrap gap-6 text-sm">
            <span><span className="text-white font-bold">{summary.total ?? predictions.length}</span><span className="text-gray-400"> total rows</span></span>
            <span><span className="text-green-400 font-bold">{summary.correct ?? "—"}</span><span className="text-gray-400"> correct</span></span>
            <span><span className="text-red-400 font-bold">{summary.wrong ?? "—"}</span><span className="text-gray-400"> wrong</span></span>
            <span><span className="text-purple-300 font-bold">{summary.accuracy_pct ?? "—"}%</span><span className="text-gray-400"> accuracy</span></span>
          </div>
        )}
        {isRegress && (
          <div className="flex flex-wrap gap-6 text-sm">
            <span><span className="text-white font-bold">{summary.total ?? predictions.length}</span><span className="text-gray-400"> total rows</span></span>
          </div>
        )}
        {isClustering && cluster_counts && (
          <div className="flex flex-wrap gap-4 text-sm">
            {Object.entries(cluster_counts).map(([k, v]) => (
              <span key={k}>
                <span className="text-blue-300 font-bold">Cluster {k}:</span>
                <span className="text-gray-300"> {v} rows</span>
              </span>
            ))}
          </div>
        )}
      </div>

      {/* Metric cards */}
      {numericMetrics.length > 0 && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {numericMetrics.map(([k, v]) => (
            <div key={k} className="bg-gray-800 border border-gray-700 rounded-xl p-4 text-center">
              <div className="text-xs text-gray-400 mb-1">{k.toUpperCase()}</div>
              <div className="text-xl font-bold text-purple-300">{fmtMetric(k, v)}</div>
            </div>
          ))}
        </div>
      )}

      {/* Classification report */}
      {metrics.classification_report && (
        <div className="bg-gray-900 rounded-lg p-4">
          <div className="text-gray-400 text-xs font-semibold mb-2 uppercase tracking-wide">Classification Report</div>
          <pre className="text-gray-300 text-xs font-mono overflow-x-auto whitespace-pre">{metrics.classification_report}</pre>
        </div>
      )}

      {/* Search */}
      {predictions.length > 0 && (
        <div className="flex items-center gap-3">
          <input
            type="text"
            placeholder="Search rows…"
            value={search}
            onChange={e => { setSearch(e.target.value); setPage(0); }}
            className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-brand-500 w-64"
          />
          <span className="text-gray-500 text-xs">{filtered.length} of {predictions.length} rows</span>
        </div>
      )}

      {/* Table */}
      {pageRows.length > 0 && (
        <div className="overflow-x-auto rounded-xl border border-gray-700">
          <table className="w-full text-xs">
            <thead>
              <tr className="bg-gray-900 border-b border-gray-700 text-gray-400 text-left">
                <th className="py-2 px-3 font-medium">#</th>
                {featureCols.map(c => <th key={c} className="py-2 px-3 font-medium">{c}</th>)}
                {(isClassif || isRegress) && <th className="py-2 px-3 font-medium text-purple-300">Actual</th>}
                {(isClassif || isRegress) && <th className="py-2 px-3 font-medium text-purple-300">Predicted</th>}
                {isClustering && <th className="py-2 px-3 font-medium text-blue-300">Cluster</th>}
                {isClassif   && <th className="py-2 px-3 font-medium">Result</th>}
                {isRegress   && <th className="py-2 px-3 font-medium">Error</th>}
              </tr>
            </thead>
            <tbody>
              {pageRows.map((row, i) => {
                const idx      = page * PAGE_SIZE + i;
                const correct  = row.correct === true;
                const wrong    = row.correct === false;
                return (
                  <tr key={idx} className={`border-b border-gray-800 ${wrong ? "bg-red-950/10" : correct ? "bg-green-950/10" : ""}`}>
                    <td className="py-1.5 px-3 text-gray-500">{idx + 1}</td>
                    {featureCols.map(c => (
                      <td key={c} className="py-1.5 px-3 text-gray-300 max-w-24 truncate" title={String(row[c] ?? "")}>
                        {String(row[c] ?? "—")}
                      </td>
                    ))}
                    {(isClassif || isRegress) && (
                      <>
                        <td className="py-1.5 px-3 text-gray-100">{String(row.actual ?? "—")}</td>
                        <td className="py-1.5 px-3 text-gray-100">{String(row.predicted ?? "—")}</td>
                      </>
                    )}
                    {isClustering && (
                      <td className="py-1.5 px-3 text-blue-300 font-mono">Cluster {row.cluster}</td>
                    )}
                    {isClassif && (
                      <td className="py-1.5 px-3">
                        <span className={`px-2 py-0.5 rounded-full text-xs font-semibold ${
                          correct ? "bg-green-900/50 text-green-400" : "bg-red-900/50 text-red-400"
                        }`}>
                          {correct ? "✓ Correct" : "✗ Wrong"}
                        </span>
                      </td>
                    )}
                    {isRegress && (
                      <td className="py-1.5 px-3 font-mono text-gray-300">
                        {typeof row.error === "number" ? (row.error >= 0 ? "+" : "") + row.error.toFixed(4) : "—"}
                      </td>
                    )}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between text-sm">
          <button disabled={page === 0} onClick={() => setPage(p => p - 1)}
            className="px-4 py-2 rounded-lg bg-gray-800 border border-gray-700 text-gray-300 disabled:opacity-40 hover:bg-gray-700 transition-colors">
            ← Prev
          </button>
          <span className="text-gray-400">Page {page + 1} of {totalPages}</span>
          <button disabled={page >= totalPages - 1} onClick={() => setPage(p => p + 1)}
            className="px-4 py-2 rounded-lg bg-gray-800 border border-gray-700 text-gray-300 disabled:opacity-40 hover:bg-gray-700 transition-colors">
            Next →
          </button>
        </div>
      )}
    </div>
  );
}

const API = "http://localhost:8000";

const STATUS_STEPS = [
  { id: "understanding", label: "Data Understanding" },
  { id: "analysis",      label: "Data Analysis"      },
  { id: "approval",      label: "Human Approval"     },
  { id: "ml",            label: "ML Engineering"     },
  { id: "evaluation",    label: "Evaluation"         },
];

function ProgressBar({ events }) {
  let reached = 0;
  const types = events.map((e) => e.type);
  if (types.includes("completed"))                                  reached = 5;
  else if (types.includes("evaluation_done"))                       reached = 4;
  else if (types.some((t) => t === "executing" && events.findIndex(e => e.type === "algorithm_selected") >= 0)) reached = 3;
  else if (types.includes("algorithm_selected"))                    reached = 3;
  else if (types.includes("approved"))                              reached = 2;
  else if (types.includes("awaiting_approval"))                     reached = 2;
  else if (types.includes("execution_success") && types.includes("code_generated")) reached = 1;
  else if (types.includes("code_generated"))                        reached = 0;

  return (
    <div className="flex items-center gap-2 overflow-x-auto pb-1">
      {STATUS_STEPS.map((step, i) => {
        const done = i < reached;
        const active = i === reached;
        return (
          <div key={step.id} className="flex items-center gap-2 shrink-0">
            <div className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold border-2 transition-all ${
              done   ? "border-brand-500 bg-brand-600 text-white" :
              active ? "border-brand-400 bg-brand-900 text-brand-300 animate-pulse" :
                       "border-gray-700 bg-gray-800 text-gray-500"
            }`}>
              {done ? "✓" : i + 1}
            </div>
            <span className={`text-xs font-medium ${done ? "text-brand-400" : active ? "text-white" : "text-gray-500"}`}>
              {step.label}
            </span>
            {i < STATUS_STEPS.length - 1 && (
              <div className={`w-8 h-0.5 ${done ? "bg-brand-600" : "bg-gray-700"}`} />
            )}
          </div>
        );
      })}
    </div>
  );
}

export default function PipelineDashboard({ session, onReset }) {
  const { sessionId, taskType, targetColumn } = session;
  const [events, setEvents] = useState([]);
  const [pipelineStatus, setPipelineStatus] = useState("running");
  const [evaluation, setEvaluation] = useState({});
  const [algorithmInfo, setAlgorithmInfo] = useState({});
  const [analysisData, setAnalysisData] = useState(null);
  const [analysisReady, setAnalysisReady] = useState(false);
  const [understandingData, setUnderstandingData] = useState(null);
  const [profilingData, setProfilingData] = useState(null);
  const [understandingReady, setUnderstandingReady] = useState(false);
  const [validationResults, setValidationResults] = useState(null);
  const [tokenUsage, setTokenUsage] = useState(null);
  const [feedback, setFeedback] = useState("");
  const [approving, setApproving] = useState(false);
  const [activeTab, setActiveTab] = useState("console");
  const esRef = useRef(null);

  const handleEvent = (evt) => {
    setEvents((prev) => [...prev, evt]);
    if (evt.type === "awaiting_approval") setPipelineStatus("awaiting_approval");
    if (evt.type === "understanding_data" && evt.data) {
      setUnderstandingData(evt.data);
      setUnderstandingReady(true);
    }
    if (evt.type === "profiling_data" && evt.data) {
      setProfilingData(evt.data);
      setUnderstandingReady(true);
    }
    if (evt.type === "analysis_data" && evt.data) {
      setAnalysisData(evt.data);
      setAnalysisReady(true);
      setActiveTab("analysis");
    }
    if (evt.type === "completed") { setPipelineStatus("completed"); esRef.current?.close(); }
    if (evt.type === "error") { setPipelineStatus("error"); esRef.current?.close(); }
    if (evt.type === "stream_end") esRef.current?.close();
    if (evt.type === "evaluation_done" && evt.data) {
      setEvaluation(evt.data);
      setActiveTab("evaluation");
    }
    if (evt.type === "validation_results" && evt.data) setValidationResults(evt.data);
    if (evt.type === "token_usage" && evt.data) setTokenUsage(evt.data);
    if (evt.type === "algorithm_selected" && evt.data) setAlgorithmInfo(evt.data);
  };

  useEffect(() => {
    const es = new EventSource(`${API}/stream/${sessionId}`);
    esRef.current = es;
    es.onmessage = (e) => { try { handleEvent(JSON.parse(e.data)); } catch (_) {} };
    es.onerror = () => { setPipelineStatus((s) => s === "running" ? "error" : s); es.close(); };
    return () => es.close();
  }, [sessionId]);

  const handleApprove = async () => {
    setApproving(true);
    try {
      const res = await fetch(`${API}/approve/${sessionId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ feedback }),
      });
      if (!res.ok) throw new Error(await res.text());
      setPipelineStatus("running");

      const es = new EventSource(`${API}/stream/${sessionId}`);
      esRef.current = es;
      es.onmessage = (e) => { try { handleEvent(JSON.parse(e.data)); } catch (_) {} };
      es.onerror = () => es.close();
    } catch (err) {
      console.error(err);
    } finally {
      setApproving(false);
    }
  };

  const isRunning = pipelineStatus === "running";
  const awaitingApproval = pipelineStatus === "awaiting_approval";

  const TABS = [
    { id: "console",    label: "Console" },
    { id: "profile",    label: "Data Profile",   badge: understandingReady },
    { id: "analysis",   label: "Analysis",        badge: analysisReady },
    { id: "code",       label: "Generated Code" },
    { id: "evaluation", label: "Evaluation",      badge: Object.keys(evaluation).length > 0 },
    { id: "validation", label: "Validation",      badge: !!validationResults },
    { id: "tokens",     label: "Token Usage",     badge: !!tokenUsage },
  ];

  return (
    <div className="space-y-6">
      {/* Header row */}
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white">Pipeline Running</h1>
          <p className="text-gray-400 text-sm mt-0.5">
            {taskType.replace(/_/g, " ")} {targetColumn ? `· target: ${targetColumn}` : ""}
          </p>
        </div>
        <button
          onClick={onReset}
          className="text-sm text-gray-400 hover:text-white border border-gray-600 hover:border-gray-400 px-4 py-2 rounded-lg transition-colors"
        >
          ← New Pipeline
        </button>
      </div>

      {/* Progress */}
      <div className="bg-gray-900 border border-gray-700 rounded-xl px-4 py-3">
        <ProgressBar events={events} />
      </div>

      {/* Approval gate */}
      {awaitingApproval && (
        <div className="bg-orange-950/40 border border-orange-700 rounded-xl p-5 space-y-4">
          <div>
            <div className="text-orange-300 font-semibold text-base">
              Data analysis complete — review results before proceeding
            </div>
            <div className="text-orange-400 text-sm mt-1">
              Optionally leave feedback for the ML agents, then approve to start modeling.
            </div>
          </div>

          {/* Structured analysis results */}
          {analysisData && (
            <div className="bg-gray-900/60 border border-orange-900/50 rounded-lg p-4">
              <AnalysisViewer data={analysisData} />
            </div>
          )}

          {/* Feedback textarea */}
          <div>
            <label className="block text-sm font-medium text-orange-300 mb-1.5">
              Feedback for ML agents <span className="text-gray-500 font-normal">(optional)</span>
            </label>
            <textarea
              value={feedback}
              onChange={(e) => setFeedback(e.target.value)}
              placeholder="e.g. Use mean instead of median for imputation, try XGBoost, focus on recall over precision..."
              rows={3}
              className="w-full bg-gray-900 border border-orange-800/60 rounded-lg px-3 py-2 text-white placeholder-gray-600 text-sm focus:outline-none focus:border-orange-600 resize-none"
            />
          </div>

          <button
            onClick={handleApprove}
            disabled={approving}
            className="w-full bg-orange-600 hover:bg-orange-500 disabled:opacity-50 text-white font-semibold px-6 py-3 rounded-lg transition-colors"
          >
            {approving ? "Resuming…" : "✓ Proceed to Modeling"}
          </button>
        </div>
      )}

      {/* Status banners */}
      {pipelineStatus === "completed" && (
        <div className="bg-green-950/40 border border-green-700 rounded-xl p-4 text-green-300 font-semibold text-center">
          Pipeline completed successfully!
        </div>
      )}
      {pipelineStatus === "error" && (
        <div className="bg-red-950/40 border border-red-700 rounded-xl p-4 text-red-300 font-semibold text-center">
          Pipeline encountered a fatal error. Check the console for details.
        </div>
      )}

      {/* Tab bar */}
      <div className="flex gap-1 border-b border-gray-700">
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setActiveTab(t.id)}
            className={`relative px-4 py-2 text-sm font-medium rounded-t-lg transition-colors ${
              activeTab === t.id
                ? "text-brand-400 border-b-2 border-brand-500 bg-gray-900"
                : "text-gray-400 hover:text-gray-200"
            }`}
          >
            {t.label}
            {t.badge && activeTab !== t.id && (
              <span className="absolute top-1.5 right-1.5 w-1.5 h-1.5 rounded-full bg-brand-400" />
            )}
          </button>
        ))}
      </div>

      {/* Tab content */}
      {activeTab === "console" && (
        <AgentConsole events={events} isRunning={isRunning} />
      )}
      {activeTab === "profile" && (
        <UnderstandingTab
          understandingData={understandingData}
          profilingData={profilingData}
          taskType={taskType}
          ready={understandingReady}
        />
      )}
      {activeTab === "analysis" && (
        <AnalysisTab
          sessionId={sessionId}
          analysisData={analysisData}
          ready={analysisReady}
        />
      )}
      {activeTab === "code" && (
        <CodeViewer sessionId={sessionId} />
      )}
      {activeTab === "evaluation" && (
        Object.keys(evaluation).length > 0 ? (() => {
          const best       = evaluation.best_model || {};
          const rawMetrics = best.metrics || {};
          const primaryKey = best.primary_metric || evaluation.primary_metric || "score";
          const primaryVal = rawMetrics[primaryKey] ?? evaluation.score;
          const isPass     = evaluation.verdict === "pass";
          const SKIP       = new Set(["algorithm","iteration","task_type","model_path","train_samples","test_samples","hyperparameters","strategy"]);
          const secondaryMetrics = Object.entries(rawMetrics).filter(([k,v]) => k !== primaryKey && !SKIP.has(k) && typeof v === "number");
          const history = evaluation.optimization_history || [];

          return (
            <div className="space-y-6">
              {/* Header */}
              <div className="flex items-center justify-between flex-wrap gap-3">
                <div>
                  <h2 className="text-xl font-bold text-white">{best.algorithm || algorithmInfo?.algorithm || "Model"} Results</h2>
                  <p className="text-gray-400 text-sm mt-0.5">{evaluation.summary}</p>
                </div>
                <span className={`inline-flex items-center gap-1 px-3 py-1 rounded-full text-sm font-semibold ${isPass ? "bg-green-900/50 text-green-400 border border-green-700" : "bg-red-900/50 text-red-400 border border-red-700"}`}>
                  {isPass ? "✓ PASS" : "✗ RETRY"}
                </span>
              </div>

              {/* Metrics */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="rounded-xl p-4 border border-brand-500 bg-brand-900/20">
                  <div className="text-xs text-gray-400 mb-1">{primaryKey.toUpperCase()}</div>
                  <div className="text-2xl font-bold text-brand-400">{fmtMetric(primaryKey, primaryVal)}</div>
                </div>
                {secondaryMetrics.slice(0, 3).map(([k, v]) => (
                  <div key={k} className="rounded-xl p-4 border border-gray-700 bg-gray-800">
                    <div className="text-xs text-gray-400 mb-1">{k.toUpperCase()}</div>
                    <div className="text-2xl font-bold text-white">{fmtMetric(k, v)}</div>
                  </div>
                ))}
              </div>

              {/* Strengths + Weaknesses */}
              <div className="grid md:grid-cols-2 gap-4">
                {evaluation.strengths?.length > 0 && (
                  <div className="bg-green-950/30 border border-green-800 rounded-xl p-4">
                    <div className="text-green-400 font-semibold text-sm mb-2">Strengths</div>
                    <ul className="space-y-1">
                      {evaluation.strengths.map((s, i) => (
                        <li key={i} className="text-green-300 text-sm flex gap-2"><span className="shrink-0">+</span><span>{s}</span></li>
                      ))}
                    </ul>
                  </div>
                )}
                {evaluation.weaknesses?.length > 0 && (
                  <div className="bg-red-950/30 border border-red-800 rounded-xl p-4">
                    <div className="text-red-400 font-semibold text-sm mb-2">Weaknesses</div>
                    <ul className="space-y-1">
                      {evaluation.weaknesses.map((s, i) => (
                        <li key={i} className="text-red-300 text-sm flex gap-2"><span className="shrink-0">−</span><span>{s}</span></li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>

              {/* Suggestions */}
              {evaluation.suggestions?.length > 0 && (
                <div className="bg-gray-800 border border-gray-700 rounded-xl p-4">
                  <div className="text-brand-400 font-semibold text-sm mb-2">Suggestions</div>
                  <ul className="space-y-1">
                    {evaluation.suggestions.map((s, i) => (
                      <li key={i} className="text-gray-300 text-sm flex gap-2"><span className="shrink-0 text-brand-500">→</span><span>{s}</span></li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Optimization history */}
              {history.length > 0 && (
                <div className="bg-gray-800 border border-gray-700 rounded-xl p-4">
                  <div className="text-gray-300 font-semibold text-sm mb-3">Optimization History</div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="text-gray-400 border-b border-gray-700 text-left">
                          <th className="py-2 pr-4 font-medium">Iter</th>
                          <th className="py-2 pr-4 font-medium">Strategy</th>
                          <th className="py-2 pr-4 font-medium">Algorithm</th>
                          <th className="py-2 font-medium text-right">{primaryKey.toUpperCase()}</th>
                        </tr>
                      </thead>
                      <tbody>
                        {history.map((h, i) => {
                          const isBest = h.algorithm === best.algorithm && h.primary_score === best.primary_score;
                          return (
                            <tr key={i} className={`border-b border-gray-800 ${isBest ? "text-yellow-300" : "text-gray-300"}`}>
                              <td className="py-2 pr-4">{h.iteration}</td>
                              <td className="py-2 pr-4 text-xs text-gray-400">{h.strategy}</td>
                              <td className="py-2 pr-4 font-mono text-xs">{h.algorithm}{isBest ? " ★" : ""}</td>
                              <td className="py-2 text-right">{typeof h.primary_score === "number" ? h.primary_score.toFixed(4) : "—"}</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {/* Actions */}
              <div className="flex flex-wrap gap-3">
                <a href={`http://localhost:8000/download/${sessionId}/model`} download="model.pkl"
                  className="inline-flex items-center gap-2 bg-brand-600 hover:bg-brand-700 text-white font-semibold px-6 py-3 rounded-lg transition-colors">
                  ⬇ Download model.pkl
                </a>
              </div>
            </div>
          );
        })() : (
          <div className="text-gray-500 text-sm italic text-center py-12">
            Evaluation results will appear here after the pipeline completes.
          </div>
        )
      )}
      {activeTab === "validation" && (
        <ValidationTab results={validationResults} />
      )}
      {activeTab === "tokens" && (
        tokenUsage ? (
          <div className="bg-gray-900 rounded-xl border border-gray-700 p-6 space-y-6">
            {/* Totals */}
            <div className="grid grid-cols-3 gap-4">
              {[
                { label: "Input Tokens",  value: tokenUsage.input_tokens  ?? 0 },
                { label: "Output Tokens", value: tokenUsage.output_tokens ?? 0 },
                { label: "Total Tokens",  value: tokenUsage.total_tokens  ?? 0 },
              ].map(({ label, value }) => (
                <div key={label} className="bg-gray-800 border border-gray-700 rounded-xl p-4 text-center">
                  <div className="text-xs text-gray-400 mb-1">{label}</div>
                  <div className="text-2xl font-bold text-white">{value.toLocaleString()}</div>
                </div>
              ))}
            </div>

            {/* Per-agent breakdown */}
            {Object.keys(tokenUsage.by_agent || {}).length > 0 && (
              <div className="overflow-x-auto">
                <div className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-3">By Agent</div>
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-gray-400 border-b border-gray-700 text-left">
                      <th className="py-2 pr-4 font-medium">Agent</th>
                      <th className="py-2 pr-4 font-medium text-right">Input</th>
                      <th className="py-2 pr-4 font-medium text-right">Output</th>
                      <th className="py-2 pr-4 font-medium text-right">Total</th>
                      <th className="py-2 font-medium text-right">Calls</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(tokenUsage.by_agent).map(([name, s]) => (
                      <tr key={name} className="border-b border-gray-800 text-gray-300">
                        <td className="py-2 pr-4 font-mono text-xs">{name}</td>
                        <td className="py-2 pr-4 text-right">{s.input.toLocaleString()}</td>
                        <td className="py-2 pr-4 text-right">{s.output.toLocaleString()}</td>
                        <td className="py-2 pr-4 text-right">{s.total.toLocaleString()}</td>
                        <td className="py-2 text-right">{s.calls}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        ) : (
          <div className="text-gray-500 text-sm italic text-center py-12">
            Token usage will appear here after the pipeline completes.
          </div>
        )
      )}
    </div>
  );
}
