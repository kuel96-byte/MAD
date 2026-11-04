import React, { useEffect, useState } from 'react';
import { api } from '../api.js';

const EMPTY = {
  name: '', source: 'manual', severity: 5, confidence: 0.5, description: '',
  tags: '', indicators: '', techniques: '',
  open_ports: 4, unpatched_cves: 8, encryption_coverage: 0.5, audit_frequency: 2,
};

export default function Threats({ role }) {
  const [list, setList] = useState([]);
  const [form, setForm] = useState(EMPTY);
  const [detail, setDetail] = useState(null);
  const [error, setError] = useState('');
  const canWrite = role === 'admin' || role === 'analyst';

  const load = () => api.threats().then(setList).catch((e) => setError(e.message));
  useEffect(() => { load(); }, []);

  function set(k, v) { setForm((f) => ({ ...f, [k]: v })); }

  async function submit(e) {
    e.preventDefault();
    setError('');
    const body = {
      name: form.name, source: form.source, severity: Number(form.severity),
      confidence: Number(form.confidence), description: form.description,
      tags: form.tags.split(',').map((s) => s.trim()).filter(Boolean),
      indicators: form.indicators.split(',').map((s) => s.trim()).filter(Boolean)
        .map((v) => ({ type: v.includes('.') && !v.includes('/') ? 'ipv4' : 'domain', value: v })),
      techniques: form.techniques.split(',').map((s) => s.trim()).filter(Boolean),
      open_ports: Number(form.open_ports), unpatched_cves: Number(form.unpatched_cves),
      encryption_coverage: Number(form.encryption_coverage), audit_frequency: Number(form.audit_frequency),
    };
    try {
      await api.createThreat(body);
      setForm(EMPTY);
      load();
    } catch (err) { setError(err.message); }
  }

  async function open(id) {
    setError('');
    try { setDetail(await api.threat(id)); } catch (e) { setError(e.message); }
  }

  async function remove(id) {
    if (!confirm('Удалить артефакт?')) return;
    await api.deleteThreat(id);
    if (detail && detail.id === id) setDetail(null);
    load();
  }

  return (
    <div className="two-col">
      <section>
        <h2>Артефакты threat intelligence</h2>
        {error && <div className="alert">{error}</div>}
        <table className="table">
          <thead><tr><th>Название</th><th>Severity</th><th>Теги</th><th>Q-score</th><th /></tr></thead>
          <tbody>
            {list.map((t) => (
              <tr key={t.id}>
                <td><button className="link" onClick={() => open(t.id)}>{t.name}</button></td>
                <td><span className={`sev sev-${t.severity >= 8 ? 'high' : t.severity >= 6 ? 'mid' : 'low'}`}>{t.severity}</span></td>
                <td>{t.tags.map((x) => <span key={x} className="chip">{x}</span>)}</td>
                <td>{t.quantum_score ?? '—'}</td>
                <td>{role === 'admin' && <button className="btn danger sm" onClick={() => remove(t.id)}>✕</button>}</td>
              </tr>
            ))}
          </tbody>
        </table>

        {canWrite && (
          <form className="card form" onSubmit={submit}>
            <h3>Новый индикатор / артефакт</h3>
            <label>Название<input required value={form.name} onChange={(e) => set('name', e.target.value)} /></label>
            <label>Описание<textarea rows="2" value={form.description} onChange={(e) => set('description', e.target.value)} /></label>
            <div className="row">
              <label>Severity (1–10)<input type="number" min="1" max="10" value={form.severity} onChange={(e) => set('severity', e.target.value)} /></label>
              <label>Confidence<input type="number" step="0.05" min="0" max="1" value={form.confidence} onChange={(e) => set('confidence', e.target.value)} /></label>
              <label>Открытые порты<input type="number" min="0" value={form.open_ports} onChange={(e) => set('open_ports', e.target.value)} /></label>
              <label>Незакрытые CVE<input type="number" min="0" value={form.unpatched_cves} onChange={(e) => set('unpatched_cves', e.target.value)} /></label>
            </div>
            <div className="row">
              <label>Шифрование (0–1)<input type="number" step="0.05" min="0" max="1" value={form.encryption_coverage} onChange={(e) => set('encryption_coverage', e.target.value)} /></label>
              <label>Аудитов/час<input type="number" min="0" value={form.audit_frequency} onChange={(e) => set('audit_frequency', e.target.value)} /></label>
            </div>
            <label>Теги (через запятую)<input value={form.tags} onChange={(e) => set('tags', e.target.value)} placeholder="powershell, cpu" /></label>
            <label>Индикаторы IOC<input value={form.indicators} onChange={(e) => set('indicators', e.target.value)} placeholder="185.220.101.4, c2.example" /></label>
            <label>Техники ATT&amp;CK<input value={form.techniques} onChange={(e) => set('techniques', e.target.value)} placeholder="T1059.001" /></label>
            <button className="btn primary">Добавить и рассчитать квантовый профиль</button>
          </form>
        )}
      </section>

      <aside className="card detail">
        <h3>Квантовый профиль артефакта</h3>
        {!detail ? <div className="muted">Выберите артефакт в таблице.</div> : (
          <div>
            <h4>{detail.name}</h4>
            <p className="muted">{detail.description}</p>
            <pre className="json">{JSON.stringify(detail.quantum_profile, null, 2)}</pre>
            <h4>Техники MITRE ATT&amp;CK</h4>
            <ul>{detail.attack_techniques.map((t) => (
              <li key={t.id}><b>{t.id}</b> — {t.name} <span className="muted">({t.cpu_level})</span></li>))}
            </ul>
          </div>
        )}
      </aside>
    </div>
  );
}
