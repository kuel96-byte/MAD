import React, { useEffect, useState } from 'react';
import { api } from '../api.js';

const CPUS = [
  'Intel Core i7 (Skylake-class)',
  'Intel Xeon Gold (Spectre-variant2 fixed)',
  'AMD Ryzen 7 (Zen 3)',
  'Apple M2 (ARM, PAC enabled)',
];

export default function ComputerModel() {
  const [cpu, setCPU] = useState(CPUS[0]);
  const [model, setModel] = useState(null);
  const [error, setError] = useState('');

  const load = (name) =>
    api.computerModel(name).then(setModel).catch((e) => setError(e.message));
  useEffect(() => { load(cpu); }, [cpu]);

  if (error) return <div className="alert">{error}</div>;
  if (!model) return <div className="muted">Построение модели…</div>;

  return (
    <div>
      <h2>Компьютерная модель через квантовые модели CPU</h2>
      <label>Процессор:
        <select value={cpu} onChange={(e) => setCPU(e.target.value)}>
          {CPUS.map((c) => <option key={c} value={c}>{c}</option>)}
        </select>
      </label>

      <div className="cards">
        <div className="card">
          <h4>Классическая вычислительная модель</h4>
          <ul>
            <li>{model.classical_model.type}</li>
            <li>ISA: {model.classical_model.isa}, слово {model.classical_model.word_size_bits} бит</li>
            <li>Конвейер: {model.classical_model.stages.join(' → ')} ({model.classical_model.pipeline_depth} стадий)</li>
            <li>Адресное пространство: {model.classical_model.address_space}</li>
            <li className="muted">{model.classical_model.states}</li>
          </ul>
        </div>
        <div className="card violet-border">
          <h4>⚛ Квантовое представление</h4>
          <ul>
            {Object.entries(model.quantum_analogy).map(([k, v]) => (
              <li key={k}><b>{k.replace(/_/g, ' ')}:</b> {v}</li>
            ))}
          </ul>
        </div>
      </div>

      <h3>Квантовое блуждание по конвейеру (Hadamard diffusion)</h3>
      <div className="walk">
        {model.quantum_pipeline_walk.probabilities.map((p, i) => (
          <div key={i} className="walk-bar" title={`стадия ${i}: ${p}`}>
            <div className="walk-fill" style={{ height: `${Math.min(100, p * 400)}%` }} />
            <span>{i}</span>
          </div>
        ))}
      </div>
      <p className="muted">pipeline_spread = {model.quantum_pipeline_walk.pipeline_spread} —
        средняя «глубина проникновения» амплитуды вредоносного кода по стадиям конвейера.</p>

      <h3>Уровни безопасности модели</h3>
      <div className="cards">
        {model.security_layers.map((l) => (
          <div key={l.layer} className="card">
            <h4>{l.layer}</h4>
            <p><b>Риски:</b> {l.risks.join(', ')}</p>
            <p><b>Митигации:</b> {l.mitigations.join(', ')}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
