import { useState, useCallback, useMemo } from "react";
import { Sidebar, SensorChart, ChatWindow } from "../components/dashboard";
import { useInference } from "../hooks/useInference";
import {
  generateChartData,
  identifyAnomalyRegions,
  getWindowReadings,
  ENGINE_PROFILES,
  getEngineProfile,
} from "../data/testData";
import type { AnomalyRegion } from "../types/api";

const WINDOW_SIZE = 10;
const DEFAULT_THRESHOLD = 0.12;
const DEFAULT_ENGINE = ENGINE_PROFILES[0].id;

export function DashboardPage() {
  const [threshold, setThreshold] = useState(DEFAULT_THRESHOLD);
  const [selectedEngineId, setSelectedEngineId] = useState(DEFAULT_ENGINE);
  const [selectedRegion, setSelectedRegion] = useState<AnomalyRegion | null>(
    null,
  );
  const { predict, loading, error, result, reset } = useInference();

  const currentEngine = getEngineProfile(selectedEngineId);

  const chartData = useMemo(
    () => generateChartData(currentEngine.totalCycles, selectedEngineId),
    [selectedEngineId, currentEngine.totalCycles],
  );
  const anomalyRegions = useMemo(
    () => identifyAnomalyRegions(chartData, 0.05),
    [chartData],
  );

  const handleRegionClick = useCallback(
    async (start: number, end: number) => {
      setSelectedRegion({ start, end });
      reset();

      const readings = getWindowReadings(end, WINDOW_SIZE, selectedEngineId);

      if (readings.length < WINDOW_SIZE) {
        return;
      }

      await predict({
        sensor_readings: readings,
        window_size: WINDOW_SIZE,
        threshold,
      });
    },
    [predict, reset, threshold, selectedEngineId],
  );

  const handleEngineChange = useCallback(
    (engineId: string) => {
      setSelectedEngineId(engineId);
      setSelectedRegion(null);
      reset();
    },
    [reset],
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
        engineId={selectedEngineId}
        onEngineChange={handleEngineChange}
        engines={ENGINE_PROFILES}
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
        threshold={result?.threshold ?? threshold}
      />
    </div>
  );
}
