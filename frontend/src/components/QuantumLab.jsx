import React, { useState } from 'react';
import { api } from '../api.js';

export default function QuantumLab() {
  const [params, setParams] = useState({
    severity: 7, confidence: 0.6, open_ports: 5, unpatched_cves: 12,
    encryption_coverage: 0.4, audit_frequency: 3,
    cpu_name: 'Intel Core i7 (Skylake-class)',
  });
  const [profile, setProfile] = useState(null);
  const [nc, setNc] = useState({ image_hash: 'deadbeef', reference_hash: 'deadbeef' });
  const [ncResult, setNcResult] = useState(null);
  const [error, setError] = useState('');

  const set = (k, v) => setParams((p) => ({ ...p, [k]: v }));

  async function run() {
    setError('');
    try {
      setProfile(await api.quantumProfile({
        ...params,
        severity: Number(params.severity), confidence: Number(params.confidence),
        open_ports: Number(params.open_ports), unpatched_cves: Number(params.unpatched_cves),
        encryption_coverage: Number(params.encryption_coverage), audit_frequency: Number(params.audit_frequency),
      }));
    } catch (e) { setError(e.message); }
  }

  async function runNoCloning() {
    setError('');
    try { setNcResult(await api.noCloning(nc)); } catch (e) { setError(e.message); }
  }

  return (
    <div className="two-col">
      <section>
        <h2>Квантовые модели компьютерной безопасности</h2>
        {error && <div className="alert">{error}</div>}

        <form className="card form" onSubmit={(e) => { e.preventDefault(); run(); }}>
          <h3>Параметры CPU-модели и защищённости</h3>
          <label>Severity угрозы (1–10)
            <input type="range" min="1" max="10" value={params.severity}
                   onChange={(e) => set('severity', e.target.value)} /> {params.severity}
          </label>
          <label>Confidence
            <input type="range" min="0" max="1" step="0.05" value={params.confidence}
                   onChange={(e) => set('confidence', e.target.value)} /> {params.confidence}
          </label>
          <div className="row">
            <label>Открытые порты<input type="number" min="0" value={params.open_ports} onChange={(e) => set('open_ports', e.target.value)} /></label>
            <label>Незакрытые CVE<input type="number" min="0" value={params.unpatched_cves} onChange={(e) => set('unpatched_cves', e.target.value)} /></label>
            <label>Шифрование (0–1)<input type="number" step="0.05" min="0" max="1" value={params.encryption_coverage} onChange={(e) => set('encryption_coverage', e.target.value)} /></label>
            <label>Аудитов/час<input type="number" min="0" value={params.audit_frequency} onChange={(e) => set('audit_frequency', e.target.value)} /></label>
          </div>
          <button className="btn primary">Рассчитать квантовый профиль</button>
        </form>

        <form className="card form" onSubmit={(e) => { e.preventDefault(); runNoCloning(); }}>
          <h3>No-Cloning аудит целостности образа</h3>
          <div className="row">
            <label>Хэш образа<input value={nc.image_hash} onChange={(e) => setNc((s) => ({ ...s, image_hash: e.target.value }))} /></label>
            <label>Эталонный хэш<input value={nc.reference_hash} onChange={(e) => setNc((s) => ({ ...s, reference_hash: e.target.value }))} /></label>
          </div>
          <button className="btn primary">Проверить клонирование/подмену</button>
          {ncResult && (
            <pre className="json">{JSON.stringify(ncResult, null, 2)}</pre>
          )}
        </form>
      </section>

      <aside className="card detail violet-border">
        <h3>Результат моделирования</h3>
        {!profile ? <div className="muted">Запустите расчёт.</div> : (
          <div>
            <div className="qscore">
              Composite: <b>{profile.composite_score}</b>
              <div className="bar"><div style={{ width: `${profile.composite_score * 100}%` }} /></div>
            </div>
            <h4>|ψ⟩ = α|safe⟩ + β|threat⟩</h4>
            <p>α = {profile.qubit_state.alpha}, β = {profile.qubit_state.beta},
               P(threat) = |β|² = <b>{profile.qubit_state.p_threat}</b></p>
            <h4>Отжиг QUBO (выбор конфигурации защиты)</h4>
            <ul>{Object.entries(profile.annealing.configuration).map(([k, v]) => (
              <li key={k}>{v ? '✅' : '⬜'} {k.replace(/_/g, ' ')}</li>))}
            </ul>
            <p>ground energy = {profile.annealing.ground_energy},
               quantum_vulnerability = <b>{profile.annealing.quantum_vulnerability}</b></p>
            <h4>Зенов эффект (частые аудиты подавляют эскалацию)</h4>
            <p>survival = {profile.zeno.survival_probability}, gain = {profile.zeno.mitigation_gain}</p>
            <details><summary>Полный JSON</summary>
              <pre className="json">{JSON.stringify(profile, null, 2)}</pre>
            </details>
          </div>
        )}
      </aside>
    </div>
  );
}
