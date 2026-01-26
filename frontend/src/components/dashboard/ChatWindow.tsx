import { Bot, AlertTriangle } from "lucide-react";

interface Props {
  diagnosis: string | null;
  loading: boolean;
  error: string | null;
  anomalyScore: number | null;
  sensorContributions: Record<string, number> | null;
  threshold: number | null;
}

export function ChatWindow({
  diagnosis,
  loading,
  error,
  anomalyScore,
  sensorContributions,
  threshold,
}: Props) {
  return (
    <aside className="chat-window">
      <div className="chat-header">
        <Bot size={20} />
        <h3>AI Diagnostic Copilot</h3>
      </div>

      <div className="chat-content">
        {loading && (
          <div className="chat-bubble loading">
            <div className="loading-dots">
              <span></span>
              <span></span>
              <span></span>
            </div>
            <span>Analyzing anomaly...</span>
          </div>
        )}

        {error && (
          <div className="chat-bubble error">
            <AlertTriangle size={16} />
            <div>
              <strong>Error</strong>
              <p>{error}</p>
            </div>
          </div>
        )}

        {!loading && !error && !diagnosis && (
          <div className="chat-bubble hint">
            <p>
              Click a red anomaly region on the chart to get an AI-powered
              diagnosis.
            </p>
            <p className="hint-sub">
              The AI will analyze sensor patterns and provide maintenance
              recommendations.
            </p>
          </div>
        )}

        {anomalyScore !== null && sensorContributions && (
          <div className="chat-bubble stats">
            <div className="stat-row">
              <span>Anomaly Score</span>
              <span
                className={
                  threshold !== null && anomalyScore > threshold
                    ? "high"
                    : "normal"
                }
              >
                {anomalyScore.toFixed(4)}
              </span>
            </div>
            <div className="stat-row">
              <span>Top Contributors</span>
            </div>
            <ul className="contributors">
              {Object.entries(sensorContributions).map(([sensor, pct]) => (
                <li key={sensor}>
                  <span className="sensor-name">{sensor}</span>
                  <span className="sensor-pct">{(pct * 100).toFixed(1)}%</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {diagnosis && (
          <div className="chat-bubble diagnosis">
            <div className="diagnosis-label">Diagnosis</div>
            <p>{diagnosis}</p>
          </div>
        )}
      </div>
    </aside>
  );
}
