// Простая обёртка над fetch с JWT-токеном из sessionStorage.
const TOKEN_KEY = 'tiq_token';
const USER_KEY = 'tiq_user';

export function getToken() { return sessionStorage.getItem(TOKEN_KEY); }
export function getUser() {
  const raw = sessionStorage.getItem(USER_KEY);
  return raw ? JSON.parse(raw) : null;
}
export function setSession(token, user) {
  sessionStorage.setItem(TOKEN_KEY, token);
  sessionStorage.setItem(USER_KEY, JSON.stringify(user));
}
export function clearSession() {
  sessionStorage.removeItem(TOKEN_KEY);
  sessionStorage.removeItem(USER_KEY);
}

async function request(path, options = {}) {
  const headers = { 'Content-Type': 'application/json', ...(options.headers || {}) };
  const token = getToken();
  if (token) headers.Authorization = `Bearer ${token}`;
  const resp = await fetch(path, { ...options, headers });
  if (resp.status === 401) {
    clearSession();
    window.dispatchEvent(new Event('tiq:logout'));
  }
  if (resp.status === 204) return null;
  const data = await resp.json().catch(() => null);
  if (!resp.ok) {
    const detail = data && (data.detail || data.message);
    throw new Error(typeof detail === 'string' ? detail : `Ошибка ${resp.status}`);
  }
  return data;
}

export const api = {
  login: (username, password) =>
    request('/api/auth/login', { method: 'POST', body: JSON.stringify({ username, password }) }),
  me: () => request('/api/auth/me'),
  threats: () => request('/api/threats'),
  threat: (id) => request(`/api/threats/${id}`),
  createThreat: (body) => request('/api/threats', { method: 'POST', body: JSON.stringify(body) }),
  deleteThreat: (id) => request(`/api/threats/${id}`, { method: 'DELETE' }),
  matrix: () => request('/api/mitre/matrix'),
  computerModel: (cpu) => request(`/api/computer-model?cpu_name=${encodeURIComponent(cpu)}`),
  quantumProfile: (body) => request('/api/quantum/profile', { method: 'POST', body: JSON.stringify(body) }),
  noCloning: (body) => request('/api/quantum/no-cloning', { method: 'POST', body: JSON.stringify(body) }),
  stats: () => request('/api/stats'),
};
