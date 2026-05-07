import { useState, useRef } from "react";

const API = "http://localhost:8000";

export default function UploadScreen({ onStart }) {
  const [file, setFile] = useState(null);
  const [columns, setColumns] = useState([]);
  const [taskType, setTaskType] = useState("supervised_classification");
  const [targetColumn, setTargetColumn] = useState("");
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const inputRef = useRef();

  const parseColumns = (f) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const firstLine = e.target.result.split("\n")[0] || "";
      // Handle quoted CSV headers
      const cols = firstLine.split(",").map((c) =>
        c.trim().replace(/^["']|["']$/g, "")
      );
      setColumns(cols);
      setTargetColumn("");
    };
    // Only read first 2KB — enough for headers
    reader.readAsText(f.slice(0, 2048));
  };

  const handleFileSelect = (f) => {
    if (!f?.name.endsWith(".csv")) {
      setError("Only CSV files are supported.");
      return;
    }
    setFile(f);
    setError("");
    parseColumns(f);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    handleFileSelect(e.dataTransfer.files[0]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) { setError("Please upload a CSV file."); return; }
    if (taskType !== "unsupervised" && !targetColumn) {
      setError("Please select a target column.");
      return;
    }

    setUploading(true);
    setError("");

    try {
      const form = new FormData();
      form.append("file", file);
      const uploadRes = await fetch(`${API}/upload`, { method: "POST", body: form });
      if (!uploadRes.ok) throw new Error(await uploadRes.text());
      const { session_id } = await uploadRes.json();

      const startRes = await fetch(`${API}/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id,
          task_type: taskType,
          target_column: taskType === "unsupervised" ? null : targetColumn,
        }),
      });
      if (!startRes.ok) throw new Error(await startRes.text());

      onStart({ sessionId: session_id, taskType, targetColumn });
    } catch (err) {
      setError(err.message || "Upload failed.");
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="max-w-xl mx-auto mt-12">
      <h1 className="text-3xl font-bold text-white mb-2">Upload Your Dataset</h1>
      <p className="text-gray-400 mb-8">
        Upload a CSV file and configure your ML task. The agent pipeline will handle the rest.
      </p>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Drop zone */}
        <div
          onDrop={handleDrop}
          onDragOver={(e) => e.preventDefault()}
          onClick={() => inputRef.current.click()}
          className="border-2 border-dashed border-gray-600 rounded-xl p-10 text-center cursor-pointer hover:border-brand-500 transition-colors"
        >
          <input
            ref={inputRef}
            type="file"
            accept=".csv"
            className="hidden"
            onChange={(e) => handleFileSelect(e.target.files[0])}
          />
          {file ? (
            <div>
              <div className="text-2xl mb-1">📄</div>
              <div className="text-white font-medium">{file.name}</div>
              <div className="text-gray-400 text-sm">{(file.size / 1024).toFixed(1)} KB</div>
              {columns.length > 0 && (
                <div className="text-gray-500 text-xs mt-1">{columns.length} columns detected</div>
              )}
            </div>
          ) : (
            <div>
              <div className="text-4xl mb-3 text-gray-500">⬆</div>
              <div className="text-gray-300 font-medium">Drop CSV here or click to browse</div>
              <div className="text-gray-500 text-sm mt-1">Only .csv files supported</div>
            </div>
          )}
        </div>

        {/* Task type */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">Task Type</label>
          <select
            value={taskType}
            onChange={(e) => { setTaskType(e.target.value); setTargetColumn(""); }}
            className="w-full bg-gray-800 border border-gray-600 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-brand-500"
          >
            <option value="supervised_classification">Supervised — Classification</option>
            <option value="supervised_regression">Supervised — Regression</option>
            <option value="unsupervised">Unsupervised — Clustering</option>
          </select>
        </div>

        {/* Target column — dropdown if columns parsed, fallback text input */}
        {taskType !== "unsupervised" && (
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Target Column <span className="text-red-400">*</span>
            </label>
            {columns.length > 0 ? (
              <select
                value={targetColumn}
                onChange={(e) => setTargetColumn(e.target.value)}
                className="w-full bg-gray-800 border border-gray-600 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-brand-500"
              >
                <option value="">— select target column —</option>
                {columns.map((col) => (
                  <option key={col} value={col}>{col}</option>
                ))}
              </select>
            ) : (
              <input
                type="text"
                value={targetColumn}
                onChange={(e) => setTargetColumn(e.target.value)}
                placeholder="e.g. label, target, Survived"
                className="w-full bg-gray-800 border border-gray-600 rounded-lg px-3 py-2 text-white placeholder-gray-500 focus:outline-none focus:border-brand-500"
              />
            )}
          </div>
        )}

        {error && (
          <div className="bg-red-950 border border-red-700 text-red-300 px-4 py-3 rounded-lg text-sm">
            {error}
          </div>
        )}

        <button
          type="submit"
          disabled={uploading}
          className="w-full bg-brand-600 hover:bg-brand-700 disabled:opacity-50 text-white font-semibold py-3 rounded-lg transition-colors"
        >
          {uploading ? "Uploading & Starting Pipeline…" : "Start Agent Pipeline →"}
        </button>
      </form>
    </div>
  );
}
