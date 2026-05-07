import { useEffect, useRef, useState } from "react";
import AgentConsole from "./AgentConsole";
import AnalysisViewer from "./AnalysisViewer";
import CodeViewer from "./CodeViewer";
import ResultsDashboard from "./ResultsDashboard";

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
  const [feedback, setFeedback] = useState("");
  const [approving, setApproving] = useState(false);
  const [activeTab, setActiveTab] = useState("console");
  const esRef = useRef(null);

  const handleEvent = (evt) => {
    setEvents((prev) => [...prev, evt]);
    if (evt.type === "awaiting_approval") setPipelineStatus("awaiting_approval");
    if (evt.type === "analysis_data" && evt.data) setAnalysisData(evt.data);
    if (evt.type === "completed") { setPipelineStatus("completed"); esRef.current?.close(); }
    if (evt.type === "error") { setPipelineStatus("error"); esRef.current?.close(); }
    if (evt.type === "stream_end") esRef.current?.close();
    if (evt.type === "evaluation_done" && evt.data) {
      setEvaluation(evt.data);
      setActiveTab("results");
    }
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
    { id: "console", label: "Console" },
    { id: "code",    label: "Generated Code" },
    { id: "results", label: "Results" },
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
            className={`px-4 py-2 text-sm font-medium rounded-t-lg transition-colors ${
              activeTab === t.id
                ? "text-brand-400 border-b-2 border-brand-500 bg-gray-900"
                : "text-gray-400 hover:text-gray-200"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      {activeTab === "console" && (
        <AgentConsole events={events} isRunning={isRunning} />
      )}
      {activeTab === "code" && (
        <CodeViewer sessionId={sessionId} />
      )}
      {activeTab === "results" && (
        Object.keys(evaluation).length > 0 ? (
          <ResultsDashboard
            evaluation={evaluation}
            algorithmInfo={algorithmInfo}
            sessionId={sessionId}
          />
        ) : (
          <div className="text-gray-500 text-sm italic text-center py-12">
            Results will appear here after the model is evaluated.
          </div>
        )
      )}
    </div>
  );
}
