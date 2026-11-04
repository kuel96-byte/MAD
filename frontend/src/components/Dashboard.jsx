import React, { useEffect, useState } from 'react';
import { api } from '../api.js';

export default function Dashboard() {
  const [stats, setStats] = useState(null);
  const [threats, setThreats] = useState([]);
  const [error, setError] = useState('');

  useEffect(() => {
    Promise.all([api.stats(), api.threats()])
      .then(([s, t]) => { setStats(s); setThreats(t.slice(0, 5)); })
      .catch((e) => setError(e.message));
  }, []);

  if (error) return <div className="alert">{error}</div>;
  if (!stats) return <div className="muted">Загрузка…</div>;

  return (
    <div>
      <h2>Оперативная сводка</h2>
      <div className="cards">
        <div className="card stat"><span className="stat-num">{stats.total_threats}</span>артефактов TI</div>
        <div className="card stat"><span className="stat-num red">{stats.high_severity}</span>критических (severity ≥ 8)</div>
        <div className="card stat"><span className="stat-num violet">{stats.avg_quantum_score}</span>средняя квантовая оценка уязвимости</div>
      </div>

      <h3>Топ угроз по severity</h3>
      <table className="table">
        <thead><tr><th>Название</th><th>Источник</th><th>Severity</th><th>Confidence</th><th>Quantum score</th></tr></thead>
        <tbody>
          {threats.map((t) => (
            <tr key={t.id}>
              <td>{t.name}</td>
              <td>{t.source}</td>
              <td><span className={`sev sev-${t.severity >= 8 ? 'high' : t.severity >= 6 ? 'mid' : 'low'}`}>{t.severity}</span></td>
              <td>{t.confidence}</td>
              <td>{t.quantum_score ?? '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
