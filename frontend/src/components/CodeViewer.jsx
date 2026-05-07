import { useState } from "react";

const API = "http://localhost:8000";

const STEPS = [
  { key: "understanding", label: "Step 1: Data Understanding" },
  { key: "analysis",      label: "Step 2: Data Analysis"      },
  { key: "ml",            label: "Step 3: ML Engineering"      },
];

export default function CodeViewer({ sessionId }) {
  const [activeStep, setActiveStep] = useState("understanding");
  const [code, setCode] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [copied, setCopied] = useState(false);

  const fetchCode = async (step) => {
    setActiveStep(step);
    if (code[step]) return;
    setLoading(true);
    setError("");
    try {
      const res = await fetch(`${API}/code/${sessionId}/${step}`);
      if (!res.ok) throw new Error((await res.json()).detail || "Not yet generated.");
      const data = await res.json();
      setCode((prev) => ({ ...prev, [step]: data.code }));
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(code[activeStep] || "");
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="bg-gray-900 rounded-xl border border-gray-700 overflow-hidden">
      <div className="flex items-center gap-1 px-4 py-3 border-b border-gray-700 bg-gray-800 overflow-x-auto">
        {STEPS.map((s) => (
          <button
            key={s.key}
            onClick={() => fetchCode(s.key)}
            className={`px-3 py-1.5 rounded-md text-xs font-medium whitespace-nowrap transition-colors ${
              activeStep === s.key
                ? "bg-brand-600 text-white"
                : "text-gray-400 hover:text-gray-200 hover:bg-gray-700"
            }`}
          >
            {s.label}
          </button>
        ))}
        {code[activeStep] && (
          <button
            onClick={handleCopy}
            className="ml-auto text-xs text-gray-400 hover:text-gray-200 px-3 py-1.5 rounded-md hover:bg-gray-700 transition-colors"
          >
            {copied ? "Copied!" : "Copy"}
          </button>
        )}
      </div>

      <div className="p-4">
        {loading && <div className="text-gray-400 text-sm italic">Loading code…</div>}
        {error && <div className="text-red-400 text-sm">{error}</div>}
        {!loading && !error && code[activeStep] && (
          <pre className="code-block">{code[activeStep]}</pre>
        )}
        {!loading && !error && !code[activeStep] && (
          <div className="text-gray-500 text-sm italic">
            Click a step above to load its generated code.
          </div>
        )}
      </div>
    </div>
  );
}
