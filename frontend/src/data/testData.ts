import type { AnomalyRegion } from "../types/api";

// Generate chart data (actual vs predicted) with degradation pattern
export function generateChartData(totalCycles: number) {
  const data = [];

  for (let i = 0; i < totalCycles; i++) {
    const degradation = Math.pow(i / totalCycles, 2);
    const noise = (Math.random() - 0.5) * 0.05;

    data.push({
      cycle: i + 1,
      actual: 0.5 + degradation * 0.4 + noise,
      predicted: 0.5 + degradation * 0.35, // Slightly behind actual (model lag)
    });
  }

  return data;
}

// Identify anomaly regions (where reconstruction error is high)
export function identifyAnomalyRegions(
  chartData: { cycle: number; actual: number; predicted: number }[],
  threshold: number = 0.05,
): AnomalyRegion[] {
  const regions: AnomalyRegion[] = [];
  let regionStart: number | null = null;

  for (const point of chartData) {
    const error = Math.abs(point.actual - point.predicted);

    if (error > threshold && regionStart === null) {
      regionStart = point.cycle;
    } else if (error <= threshold && regionStart !== null) {
      regions.push({ start: regionStart, end: point.cycle - 1 });
      regionStart = null;
    }
  }

  // Close final region if still open
  if (regionStart !== null) {
    regions.push({
      start: regionStart,
      end: chartData[chartData.length - 1].cycle,
    });
  }

  return regions;
}

// Generate synthetic sensor data with degradation pattern
function generateSensorData(totalCycles: number): number[][] {
  const data: number[][] = [];

  for (let i = 0; i < totalCycles; i++) {
    const degradation = Math.pow(i / totalCycles, 2);
    const noise = () => (Math.random() - 0.5) * 0.1;

    // 14 sensor values (matching model features)
    const reading = [
      0.5 + degradation * 0.3 + noise(),
      0.5 + noise(),
      0.5 + degradation * 0.2 + noise(),
      0.5 + degradation * 0.15 + noise(),
      0.5 + degradation * 0.4 + noise(),
      0.5 + degradation * 0.25 + noise(),
      0.5 + degradation * 0.1 + noise(),
      0.5 + degradation * 0.2 + noise(),
      0.5 + degradation * 0.35 + noise(),
      0.5 + noise(),
      0.5 + degradation * 0.15 + noise(),
      0.5 + noise(),
      0.5 + noise(),
      0.5 + degradation * 0.2 + noise(),
    ];

    data.push(reading);
  }

  return data;
}

// Get sensor readings for a window
export function getWindowReadings(
  startCycle: number,
  windowSize: number = 50,
): number[][] {
  const totalCycles = 250;
  const allData = generateSensorData(totalCycles);

  const startIdx = Math.max(0, startCycle - windowSize);
  const endIdx = Math.min(totalCycles, startCycle);

  return allData.slice(startIdx, endIdx);
}
