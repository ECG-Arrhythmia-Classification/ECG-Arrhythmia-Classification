import {CLASSES} from './signal.mjs';

export const DEFAULT_API_URL = import.meta.env?.VITE_API_URL || 'http://127.0.0.1:8000';

export function readSetting(key, fallback) {
  try { return localStorage.getItem(`ecg-studio.${key}`) || fallback; }
  catch { return fallback; }
}

export function saveSetting(key, value) {
  try { localStorage.setItem(`ecg-studio.${key}`, value); }
  catch { /* Settings still work when storage is unavailable. */ }
}

export async function fetchApiJson(url, path, options = {}) {
  let response;
  try { response = await fetch(`${url.trim().replace(/\/$/, '')}${path}`, options); }
  catch (error) {
    if (error.name === 'AbortError') throw error;
    throw new Error('Cannot connect to FastAPI. Check the backend, API URL, and CORS settings.');
  }
  let body;
  try { body = await response.json(); }
  catch { throw new Error(`The API did not return valid JSON (HTTP ${response.status}).`); }
  if (!response.ok) {
    throw new Error(typeof body.detail === 'string' ? body.detail : body.detail?.message || body.detail?.[0]?.msg || `The API rejected the request (HTTP ${response.status}).`);
  }
  return body;
}

export function modelDescriptors(body) {
  if (!Array.isArray(body.models)) throw new Error('The /models endpoint must return a model list.');
  return body.models.filter(model => model && typeof model.id === 'string').map(model => ({
    ...model,
    input_length: Number(model.input_length),
  }));
}

export function pipelineMetadata(result, descriptor) {
  const preprocessing = result?.preprocessing;
  return {
    pipeline: preprocessing?.pipeline || result?.pipeline || descriptor?.pipeline || (result?.is_demo ? 'demo' : null),
    input_length: preprocessing?.processed_length || preprocessing?.signal?.length || result?.input_length || descriptor?.input_length || null,
  };
}

export function pipelineLabel(result, descriptor) {
  const metadata = pipelineMetadata(result, descriptor);
  if (!metadata.input_length) return 'Awaiting pipeline configuration';
  return `Pipeline: ${metadata.input_length} samples · ${metadata.pipeline === 'team' ? 'Band-pass + Z-score' : 'demo'}`;
}

export function modelStatus(descriptor) {
  if (!descriptor) return 'Awaiting API configuration';
  if (descriptor.status === 'unavailable' || descriptor.status === 'error') return 'Unavailable';
  return `${descriptor.is_demo ? 'Demo' : 'Checkpoint'} · ${descriptor.input_length} samples`;
}

const classCodes = ['N', 'S', 'V', 'F', 'Q'];

function englishClass(row) {
  // Legacy API payloads may contain label_vi. Keep scores and codes intact while
  // presenting and exporting the project's consistent English class names.
  const {label_vi, ...rest} = row;
  return {...rest, label_en: CLASSES.find(item => item.code === row.code)?.label_en || row.label_en};
}

export function testExample(body) {
  if (!Array.isArray(body.signal) || body.signal.length !== 180 || !body.signal.every(Number.isFinite)
    || body.sample_rate !== 360 || body.is_synthetic !== false || !Number.isInteger(body.index)
    || !classCodes.includes(body.expected_class?.code)) {
    throw new Error('A test sample must contain 180 finite values at 360 Hz and a valid ground-truth label.');
  }
  return {...body, expected_class: englishClass(body.expected_class)};
}

export function predictionResult(body, modelId) {
  const probabilities = body.probabilities;
  if (body.model_id !== modelId || !classCodes.includes(body.prediction?.code)
    || !Number.isFinite(body.confidence) || body.confidence < 0 || body.confidence > 1
    || !Number.isFinite(body.inference_ms) || body.inference_ms < 0
    || !Array.isArray(probabilities) || probabilities.length !== classCodes.length
    || new Set(probabilities.map(p => p.code)).size !== classCodes.length
    || probabilities.some(p => !classCodes.includes(p.code) || !Number.isFinite(p.probability) || p.probability < 0 || p.probability > 1)
    || Math.abs(probabilities.reduce((sum, p) => sum + p.probability, 0) - 1) > 1e-4) {
    throw new Error('The API returned an invalid label or softmax score distribution.');
  }
  const predicted = probabilities.find(p => p.code === body.prediction.code);
  if (Math.abs(predicted.probability - body.confidence) > 1e-4 || probabilities.some(p => p.probability > body.confidence + 1e-4)) {
    throw new Error('The predicted label does not match the softmax score distribution.');
  }
  return {...body, prediction: englishClass(body.prediction), probabilities: probabilities.map(englishClass)};
}

export function resultExport(result, {source, sourceInfo, raw, descriptor, options}) {
  return {
    exported_at: new Date().toISOString(),
    source,
    input: {signal: raw, sample_rate: sourceInfo.sample_rate ?? null, is_synthetic: sourceInfo.is_synthetic, test_index: sourceInfo.index ?? null},
    expected_class: sourceInfo.expected_class ? englishClass(sourceInfo.expected_class) : null,
    matches_expected: sourceInfo.expected_class ? sourceInfo.expected_class.code === result.prediction.code : null,
    model: {id: result.model_id, name: result.model_name, backend: result.backend, is_demo: result.is_demo, checkpoint: result.checkpoint ?? descriptor?.checkpoint ?? null},
    score_meaning: result.is_demo ? 'Illustrative prototype score; not calibrated.' : 'Softmax score; not a calibrated clinical probability.',
    options: result.preprocessing?.options || options,
    result: {...result, prediction: englishClass(result.prediction), probabilities: result.probabilities.map(englishClass)},
  };
}
