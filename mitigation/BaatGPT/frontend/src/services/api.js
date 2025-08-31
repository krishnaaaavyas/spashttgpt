// frontend/src/services/api.js
const BASE = import.meta.env.VITE_API_BASE_URL || ''; // keep '' to use Vite proxy during dev

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    credentials: 'include',  // Include credentials
    ...options,
  });
  const ct = res.headers.get('content-type') || '';
  const data = ct.includes('application/json') ? await res.json() : await res.text();
  if (!res.ok) {
    const msg = typeof data === 'string' ? data.slice(0, 300) : data?.error || 'Request failed';
    throw new Error(`HTTP ${res.status}: ${msg}`);
  }
  return data;
}

// Sends user prompt -> OpenAI -> MedBERT risk (backend pipeline)
export function submitPrompt(prompt) {
  return request('/api/thread', {
    method: 'POST',
    body: JSON.stringify({ prompt }),
  });
}

// Direct risk scoring for any arbitrary text (bypasses OpenAI, hits backend proxy)
export function predictRisk(text) {
  return request('/api/predict/risk', {
    method: 'POST',
    body: JSON.stringify({ text }),
  });
}
