import React, { useEffect, useState } from 'react';
import { api } from '../api.js';

export default function MitreMatrix() {
  const [matrix, setMatrix] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => { api.matrix().then(setMatrix).catch((e) => setError(e.message)); }, []);
  if (error) return <div className="alert">{error}</div>;
  if (!matrix) return <div className="muted">Загрузка ATT&amp;CK…</div>;

  const tacticName = (id) => {
    const t = matrix.tactics.find((x) => x.id === id);
    return t ? `${t.ru} (${t.name})` : id;
  };

  return (
    <div>
      <h2>Матрица MITRE ATT&amp;CK</h2>
      <p className="muted">{matrix.matrix_version}. Техники связаны с уровнями компьютерной модели;
        в описании каждой — квантовая интерпретация перехода состояний CPU.</p>

      <div className="cards">
        {matrix.tactics.map((t) => (
          <div key={t.id} className="card tactic">
            <b>{t.name}</b><span className="muted">{t.ru}</span>
          </div>
        ))}
      </div>

      <table className="table">
        <thead><tr><th>ID</th><th>Техника</th><th>Тактика</th><th>Уровень модели</th><th>Квантовая интерпретация</th></tr></thead>
        <tbody>
          {matrix.techniques.map((tech) => (
            <tr key={tech.id}>
              <td><code>{tech.id}</code></td>
              <td>{tech.name}</td>
              <td>{tacticName(tech.tactic)}</td>
              <td><span className="chip">{tech.cpu_level}</span></td>
              <td className="muted">{tech.quantum_note}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
