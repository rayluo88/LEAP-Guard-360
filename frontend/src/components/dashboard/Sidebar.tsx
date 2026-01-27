import { Link } from "react-router-dom";
import { LayoutDashboard, BarChart3, Settings, Home } from "lucide-react";
import type { EngineProfile } from "../../data/testData";

interface Props {
  threshold: number;
  onThresholdChange: (value: number) => void;
  engineId: string;
  onEngineChange: (engineId: string) => void;
  engines: EngineProfile[];
  isAnomaly: boolean;
}

export function Sidebar({
  threshold,
  onThresholdChange,
  engineId,
  onEngineChange,
  engines,
  isAnomaly,
}: Props) {
  const currentEngine = engines.find((e) => e.id === engineId);
  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <img src="/logo-leap.png" alt="LEAP-Guard" className="sidebar-logo" />
        <h1>LEAP-Guard 360</h1>
      </div>

      <nav className="sidebar-nav">
        <Link to="/" className="nav-item">
          <Home size={20} />
          <span>Home</span>
        </Link>
        <div className="nav-item active">
          <LayoutDashboard size={20} />
          <span>Dashboard</span>
        </div>
        <div className="nav-item disabled">
          <BarChart3 size={20} />
          <span>Analytics</span>
        </div>
        <div className="nav-item disabled">
          <Settings size={20} />
          <span>Settings</span>
        </div>
      </nav>

      <div className="sidebar-section">
        <h3>Engine</h3>
        <select
          className="engine-select"
          value={engineId}
          onChange={(e) => onEngineChange(e.target.value)}
        >
          {engines.map((engine) => (
            <option key={engine.id} value={engine.id}>
              {engine.id}
            </option>
          ))}
        </select>
        {currentEngine && (
          <div className="engine-info">
            <span className="engine-aircraft">
              {currentEngine.aircraftType}
            </span>
            <span className="engine-cycles">
              {currentEngine.totalCycles} cycles
            </span>
          </div>
        )}
      </div>

      <div className="sidebar-section">
        <h3>Anomaly Threshold</h3>
        <input
          type="range"
          min="0.05"
          max="0.2"
          step="0.01"
          value={threshold}
          onChange={(e) => onThresholdChange(parseFloat(e.target.value))}
          className="threshold-slider"
        />
        <div className="threshold-labels">
          <span>Sensitive</span>
          <span className="threshold-value">{threshold.toFixed(2)}</span>
          <span>Strict</span>
        </div>
      </div>

      <div className="sidebar-status">
        <div
          className={`status-indicator ${isAnomaly ? "warning" : "healthy"}`}
        >
          <span className="status-dot"></span>
          <span className="status-text">
            {isAnomaly ? "Anomaly Detected" : "System Healthy"}
          </span>
        </div>
      </div>
    </aside>
  );
}
