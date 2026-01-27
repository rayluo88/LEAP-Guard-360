import type { AnomalyRegion } from "../types/api";

// Engine profile configuration
export interface EngineProfile {
  id: string;
  aircraftType: string;
  totalCycles: number;
  degradationRate: number; // How fast the engine degrades (0.3-0.5 typical)
  noiseLevel: number; // Sensor noise amplitude (0.03-0.08)
  modelLag: number; // How much the prediction lags behind actual (0.8-0.95)
  anomalyOnset: number; // When anomalies start appearing (0.6-0.9 of total cycles)
  seed: number; // For reproducible randomness
}

// Available engines with distinct degradation characteristics
export const ENGINE_PROFILES: EngineProfile[] = [
  {
    id: "LEAP-1A-001",
    aircraftType: "A320neo",
    totalCycles: 250,
    degradationRate: 0.4,
    noiseLevel: 0.05,
    modelLag: 0.875, // predicted = actual * 0.875
    anomalyOnset: 0.7,
    seed: 12345,
  },
  {
    id: "LEAP-1B-042",
    aircraftType: "B737 MAX 8",
    totalCycles: 300,
    degradationRate: 0.35,
    noiseLevel: 0.04,
    modelLag: 0.9,
    anomalyOnset: 0.75,
    seed: 67890,
  },
  {
    id: "LEAP-1A-117",
    aircraftType: "A321neo",
    totalCycles: 200,
    degradationRate: 0.5,
    noiseLevel: 0.06,
    modelLag: 0.85,
    anomalyOnset: 0.6,
    seed: 24680,
  },
];

// Seeded random number generator for reproducible results
function seededRandom(seed: number): () => number {
  return function () {
    seed = (seed * 9301 + 49297) % 233280;
    return seed / 233280;
  };
}

// Get engine profile by ID
export function getEngineProfile(engineId: string): EngineProfile {
  return ENGINE_PROFILES.find((e) => e.id === engineId) ?? ENGINE_PROFILES[0];
}

// Generate chart data (actual vs predicted) with engine-specific degradation pattern
export function generateChartData(
  totalCycles: number,
  engineId: string = "LEAP-1A-001",
) {
  const profile = getEngineProfile(engineId);
  const random = seededRandom(profile.seed);
  const data = [];

  for (let i = 0; i < totalCycles; i++) {
    const progress = i / totalCycles;
    const degradation = Math.pow(progress, 2) * profile.degradationRate;
    const noise = (random() - 0.5) * profile.noiseLevel * 2;

    const actual = 0.5 + degradation + noise;
    const predicted = 0.5 + degradation * profile.modelLag;

    data.push({
      cycle: i + 1,
      actual,
      predicted,
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

// Generate synthetic sensor data with engine-specific degradation pattern
function generateSensorData(
  totalCycles: number,
  engineId: string = "LEAP-1A-001",
): number[][] {
  const profile = getEngineProfile(engineId);
  const random = seededRandom(profile.seed + 1000); // Different seed for sensor data
  const data: number[][] = [];

  for (let i = 0; i < totalCycles; i++) {
    const progress = i / totalCycles;
    const degradation = Math.pow(progress, 2) * profile.degradationRate;
    const noise = () => (random() - 0.5) * profile.noiseLevel * 2;

    // 8 sensor values (matching CMAPSS model features)
    // Each engine has slightly different sensor response characteristics
    const reading = [
      0.5 + degradation * 0.75 + noise(), // N1_Fan_RPM
      0.5 + degradation * 0.1 + noise(), // T25_LPC_Temp
      0.5 + degradation * 0.5 + noise(), // N2_Core_RPM
      0.5 + degradation * 0.375 + noise(), // T30_HPC_Temp
      0.5 + degradation * 1.0 + noise(), // P30_HPC_Pressure (most affected)
      0.5 + degradation * 0.625 + noise(), // T50_LPT_Temp
      0.5 + degradation * 0.25 + noise(), // Fuel_Flow
      0.5 + degradation * 0.5 + noise(), // Vibration
    ];

    data.push(reading);
  }

  return data;
}

// Get sensor readings for a window
export function getWindowReadings(
  startCycle: number,
  windowSize: number = 10,
  engineId: string = "LEAP-1A-001",
): number[][] {
  const profile = getEngineProfile(engineId);
  const allData = generateSensorData(profile.totalCycles, engineId);

  const startIdx = Math.max(0, startCycle - windowSize);
  const endIdx = Math.min(profile.totalCycles, startCycle);

  return allData.slice(startIdx, endIdx);
}
