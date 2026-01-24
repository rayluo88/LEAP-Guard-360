import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceArea,
  ResponsiveContainer,
  Legend,
} from "recharts";
import type { AnomalyRegion, ChartData } from "../../types/api";

interface Props {
  data: ChartData[];
  anomalyRegions: AnomalyRegion[];
  onRegionClick: (start: number, end: number) => void;
  selectedRegion: AnomalyRegion | null;
}

export function SensorChart({
  data,
  anomalyRegions,
  onRegionClick,
  selectedRegion,
}: Props) {
  return (
    <div className="chart-container">
      <div className="chart-header">
        <h2>Engine Health Monitor</h2>
        <div className="chart-legend">
          <div className="legend-item">
            <span className="legend-line actual"></span>
            <span>Actual</span>
          </div>
          <div className="legend-item">
            <span className="legend-line predicted"></span>
            <span>Predicted</span>
          </div>
          <div className="legend-item">
            <span className="legend-box anomaly"></span>
            <span>Anomaly Zone</span>
          </div>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={350}>
        <LineChart
          data={data}
          margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
        >
          <XAxis
            dataKey="cycle"
            stroke="#8E8E93"
            tick={{ fill: "#8E8E93" }}
            axisLine={{ stroke: "#3A3A3A" }}
          />
          <YAxis
            stroke="#8E8E93"
            tick={{ fill: "#8E8E93" }}
            axisLine={{ stroke: "#3A3A3A" }}
            domain={[0.4, 1]}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "#2A2A2A",
              border: "1px solid #3A3A3A",
              borderRadius: "8px",
              color: "#FFFFFF",
            }}
            labelStyle={{ color: "#8E8E93" }}
          />
          <Legend wrapperStyle={{ display: "none" }} />

          {anomalyRegions.map((region, i) => (
            <ReferenceArea
              key={i}
              x1={region.start}
              x2={region.end}
              fill={
                selectedRegion?.start === region.start ? "#E85A4F" : "#E85A4F"
              }
              fillOpacity={selectedRegion?.start === region.start ? 0.4 : 0.15}
              onClick={() => onRegionClick(region.start, region.end)}
              style={{ cursor: "pointer" }}
            />
          ))}

          <Line
            type="monotone"
            dataKey="actual"
            stroke="#007AFF"
            dot={false}
            name="Actual"
            strokeWidth={2}
          />
          <Line
            type="monotone"
            dataKey="predicted"
            stroke="#32D583"
            strokeDasharray="5 5"
            dot={false}
            name="Predicted"
            strokeWidth={2}
          />
        </LineChart>
      </ResponsiveContainer>

      <p className="chart-hint">
        Click on a red anomaly region to get AI diagnosis
      </p>
    </div>
  );
}
