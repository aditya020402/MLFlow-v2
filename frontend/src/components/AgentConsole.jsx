import { useEffect, useRef } from "react";

const EVENT_STYLES = {
  agent_thinking:    { icon: "🧠", color: "text-purple-400",  bg: "bg-purple-950/40" },
  code_generated:    { icon: "📝", color: "text-blue-400",    bg: "bg-blue-950/40"   },
  executing:         { icon: "⚙️", color: "text-yellow-400",  bg: "bg-yellow-950/30" },
  execution_success: { icon: "✅", color: "text-green-400",   bg: "bg-green-950/40"  },
  execution_error:   { icon: "❌", color: "text-red-400",     bg: "bg-red-950/40"    },
  awaiting_approval: { icon: "⏸️", color: "text-orange-400", bg: "bg-orange-950/40" },
  approved:          { icon: "▶️", color: "text-green-400",   bg: "bg-green-950/40"  },
  algorithm_selected:{ icon: "🎯", color: "text-cyan-400",    bg: "bg-cyan-950/40"   },
  evaluation_done:   { icon: "📊", color: "text-teal-400",    bg: "bg-teal-950/40"   },
  completed:         { icon: "🏁", color: "text-green-300",   bg: "bg-green-950/40"  },
  error:             { icon: "💥", color: "text-red-400",     bg: "bg-red-950/40"    },
  heartbeat:         { icon: "💓", color: "text-gray-600",    bg: ""                 },
  connected:         { icon: "🔗", color: "text-gray-400",    bg: ""                 },
  stream_end:        { icon: "⏹️", color: "text-gray-400",    bg: ""                 },
};

export default function AgentConsole({ events, isRunning }) {
  const bottomRef = useRef();

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [events]);

  return (
    <div className="bg-gray-900 rounded-xl border border-gray-700 overflow-hidden">
      <div className="flex items-center gap-2 px-4 py-3 border-b border-gray-700 bg-gray-800">
        <div className="flex gap-1.5">
          <div className="w-3 h-3 rounded-full bg-red-500" />
          <div className="w-3 h-3 rounded-full bg-yellow-500" />
          <div className="w-3 h-3 rounded-full bg-green-500" />
        </div>
        <span className="text-gray-300 text-sm font-medium ml-2">Agent Console</span>
        {isRunning && (
          <span className="ml-auto flex items-center gap-1.5 text-xs text-green-400">
            <span className="pulse-dot w-2 h-2 rounded-full bg-green-400 inline-block" />
            Running…
          </span>
        )}
      </div>

      <div className="p-4 space-y-2 max-h-96 overflow-y-auto custom-scroll">
        {events.length === 0 && (
          <div className="text-gray-500 text-sm italic">Waiting for pipeline events…</div>
        )}
        {events.map((evt, i) => {
          const style = EVENT_STYLES[evt.type] || { icon: "•", color: "text-gray-400", bg: "" };
          if (evt.type === "heartbeat") return null;
          return (
            <div key={i} className={`flex gap-2 items-start rounded-lg px-3 py-2 ${style.bg}`}>
              <span className="text-base shrink-0">{style.icon}</span>
              <div className="flex-1 min-w-0">
                <span className={`text-sm ${style.color}`}>{evt.message}</span>
              </div>
            </div>
          );
        })}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
