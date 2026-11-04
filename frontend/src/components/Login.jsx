import React, { useState } from 'react';
import { api } from '../api.js';

export default function Login({ onLogin }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);

  async function submit(e) {
    e.preventDefault();
    setError(''); setBusy(true);
    try {
      const data = await api.login(username.trim(), password);
      onLogin(data);
    } catch (err) {
      setError(err.message || 'Ошибка входа');
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="login-wrap">
      <form className="login-card" onSubmit={submit}>
        <div className="login-logo">⚛</div>
        <h1>TI · ATT&amp;CK · Quantum Security</h1>
        <p className="subtitle">Платформа компьютерной безопасности.<br />Войдите в учётную запись аналитика.</p>

        <label>Логин
          <input value={username} onChange={(e) => setUsername(e.target.value)}
                 placeholder="admin" autoFocus autoComplete="username" required minLength={2} />
        </label>
        <label>Пароль
          <input type="password" value={password} onChange={(e) => setPassword(e.target.value)}
                 placeholder="••••••••" autoComplete="current-password" required minLength={4} />
        </label>

        {error && <div className="alert">{error}</div>}

        <button className="btn primary full" disabled={busy}>
          {busy ? 'Проверка…' : 'Войти'}
        </button>

        <div className="hint">Демо-доступ по умолчанию: <code>admin / admin12345</code></div>
      </form>
    </div>
  );
}
