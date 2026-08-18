/**
 * MICRONYX PS02 — Frontend API Service
 *
 * Connects to the existing FastAPI backend at http://127.0.0.1:8000
 * All requests are proxied through Vite's dev server.
 *
 * DO NOT modify the backend API contract.
 */

const BASE_URL = '';  // proxied via vite.config.js

/**
 * Check backend + model health.
 * GET /health
 * @returns {{ status: string, localizer_available: boolean, model_available: boolean }}
 */
export async function checkHealth() {
  const res = await fetch(`${BASE_URL}/health`, {
    method: 'GET',
    signal: AbortSignal.timeout(5000),
  });

  if (!res.ok) {
    throw new Error(`Health check failed: HTTP ${res.status}`);
  }

  return res.json();
}

/**
 * Run localization inference on a search/reference image pair.
 * POST /api/localize  (multipart/form-data)
 *
 * @param {File} searchFile   — 1000×1000 grayscale PNG search image
 * @param {File} referenceFile — 1000×1000 grayscale PNG reference image
 * @returns LocalizationResult
 *
 * LocalizationResult shape:
 * {
 *   success: boolean,
 *   predicted_x: number,
 *   predicted_y: number,
 *   confidence: number,       // e.g. 0.999962
 *   candidates: number,       // total candidate count
 *   runtime_ms: number,       // model runtime in ms
 *   api_wall_time_ms: number,
 *   engine: string,
 *   acquisition: {
 *     search_width: number,
 *     search_height: number,
 *     reference_width: number,
 *     reference_height: number,
 *     search_pixels_per_unit: number,
 *     reference_pixels_per_unit: number,
 *     sampling_ratio: number,
 *   },
 *   top_candidates: Array<{
 *     rank: number,
 *     x: number,
 *     y: number,
 *     probability: number,
 *     dog_score: number,
 *   }>
 * }
 */
export async function runLocalize(searchFile, referenceFile) {
  const form = new FormData();
  form.append('search', searchFile);
  form.append('reference', referenceFile);

  const res = await fetch(`${BASE_URL}/api/localize`, {
    method: 'POST',
    body: form,
    signal: AbortSignal.timeout(180_000),  // 3-minute timeout for inference
  });

  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch (_) { /* ignore */ }
    throw new Error(detail);
  }

  return res.json();
}

/**
 * Fetch a demo image from the frontend's public/demo/ directory
 * and return it as a File object for submission to /api/localize.
 *
 * @param {'search'|'reference'} which
 * @returns {Promise<File>}
 */
export async function fetchDemoFile(which) {
  const filename = which === 'search' ? 'clean_search.png' : 'clean_reference.png';
  const url = `/demo/${filename}`;

  const res = await fetch(url);
  if (!res.ok) throw new Error(`Could not load demo image: ${filename}`);

  const blob = await res.blob();
  return new File([blob], filename, { type: 'image/png' });
}
