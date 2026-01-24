// API types for LEAP-Guard 360

export interface SensorReading {
  cycle: number;
  sensors: Record<string, number>;
}

export interface PredictRequest {
  sensor_readings: number[][];
  window_size: number;
  threshold: number;
}

export interface PredictResponse {
  anomaly_score: number;
  threshold: number;
  is_anomaly: boolean;
  diagnosis: string | null;
  sensor_contributions: Record<string, number>;
}

export interface EngineData {
  engine_id: string;
  metadata: {
    aircraft_type: string;
    total_cycles: number;
  };
  cycles: SensorReading[];
}

export interface AnomalyRegion {
  start: number;
  end: number;
}

export interface ChartData {
  cycle: number;
  actual: number;
  predicted: number;
}
