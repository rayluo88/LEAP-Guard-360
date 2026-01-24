import { useState, useCallback } from "react";
import axios from "axios";
import type { PredictRequest, PredictResponse } from "../types/api";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:3000";

export function useInference() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);

  const predict = useCallback(async (request: PredictRequest) => {
    setLoading(true);
    setError(null);

    try {
      const response = await axios.post<PredictResponse>(API_URL, request);
      setResult(response.data);
      return response.data;
    } catch (err) {
      const message = err instanceof Error ? err.message : "Inference failed";
      setError(message);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  const reset = useCallback(() => {
    setResult(null);
    setError(null);
  }, []);

  return { predict, loading, error, result, reset };
}
