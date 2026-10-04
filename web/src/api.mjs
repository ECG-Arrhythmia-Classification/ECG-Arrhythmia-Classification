export const DEFAULT_API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

export function readSetting(key, fallback) {
  try { return localStorage.getItem(`ecg-studio.${key}`) || fallback; }
  catch { return fallback; }
}

export function saveSetting(key, value) {
  try { localStorage.setItem(`ecg-studio.${key}`, value); }
  catch { /* Settings still work when storage is unavailable. */ }
}

export async function fetchApiJson(url, path, options = {}) {
  const response = await fetch(`${url.trim().replace(/\/$/, '')}${path}`, options);
  let body;
  try { body = await response.json(); }
  catch { throw new Error(`API không trả về JSON hợp lệ (HTTP ${response.status}).`); }
  if (!response.ok) {
    throw new Error(typeof body.detail === 'string' ? body.detail : body.detail?.message || body.detail?.[0]?.msg || `API từ chối yêu cầu (HTTP ${response.status}).`);
  }
  return body;
}

export function modelDescriptors(body) {
  if (!Array.isArray(body.models)) throw new Error('API /models cần trả về danh sách models.');
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
  if (!metadata.input_length) return 'Chờ cấu hình pipeline';
  return `Pipeline ${metadata.input_length} mẫu · ${metadata.pipeline === 'team' ? 'Bandpass + Z-score' : 'demo'}`;
}

export function modelStatus(descriptor) {
  if (!descriptor) return 'Chờ cấu hình API';
  if (descriptor.status === 'unavailable' || descriptor.status === 'error') return 'Chưa sẵn sàng';
  return `${descriptor.is_demo ? 'Demo' : 'Checkpoint'} · ${descriptor.input_length} mẫu`;
}
