import { useState, useCallback, useMemo } from "react";
import { Sidebar, SensorChart, ChatWindow } from "../components/dashboard";
import { useInference } from "../hooks/useInference";
import {
  generateChartData,
  identifyAnomalyRegions,
  getWindowReadings,
} from "../data/testData";
import type { AnomalyRegion } from "../types/api";

const TOTAL_CYCLES = 250;
const WINDOW_SIZE = 50;

export function DashboardPage() {
  const [threshold, setThreshold] = useState(0.7);
  const [selectedRegion, setSelectedRegion] = useState<AnomalyRegion | null>(
    null,
  );
  const { predict, loading, error, result, reset } = useInference();

  const chartData = useMemo(() => generateChartData(TOTAL_CYCLES), []);
  const anomalyRegions = useMemo(
    () => identifyAnomalyRegions(chartData, 0.05),
    [chartData],
  );

  const handleRegionClick = useCallback(
    async (start: number, end: number) => {
      setSelectedRegion({ start, end });
      reset();

      const readings = getWindowReadings(end, WINDOW_SIZE);

      if (readings.length < WINDOW_SIZE) {
        return;
      }

      await predict({
        sensor_readings: readings,
        window_size: WINDOW_SIZE,
        threshold,
      });
    },
    [predict, reset, threshold],
  );

  const handleThresholdChange = useCallback(
    (value: number) => {
      setThreshold(value);
      setSelectedRegion(null);
      reset();
    },
    [reset],
  );

  const isAnomaly = anomalyRegions.length > 0;

  return (
    <div className="dashboard">
      <Sidebar
        threshold={threshold}
        onThresholdChange={handleThresholdChange}
        engineId="LEAP-1A-001"
        isAnomaly={isAnomaly}
      />

      <main className="main-content">
        <SensorChart
          data={chartData}
          anomalyRegions={anomalyRegions}
          onRegionClick={handleRegionClick}
          selectedRegion={selectedRegion}
        />
      </main>

      <ChatWindow
        diagnosis={result?.diagnosis ?? null}
        loading={loading}
        error={error}
        anomalyScore={result?.anomaly_score ?? null}
        sensorContributions={result?.sensor_contributions ?? null}
      />
    </div>
  );
}
