import { Link } from "react-router-dom";
import { LayoutDashboard, BarChart3, Settings, Home } from "lucide-react";

interface Props {
  threshold: number;
  onThresholdChange: (value: number) => void;
  engineId: string;
  isAnomaly: boolean;
}

export function Sidebar({
  threshold,
  onThresholdChange,
  engineId,
  isAnomaly,
}: Props) {
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
        <select className="engine-select" defaultValue={engineId}>
          <option value={engineId}>{engineId}</option>
        </select>
      </div>

      <div className="sidebar-section">
        <h3>Anomaly Threshold</h3>
        <input
          type="range"
          min="0.5"
          max="1.0"
          step="0.05"
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
