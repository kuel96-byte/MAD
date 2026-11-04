import React, { useEffect, useState } from 'react';
import { api, getToken, getUser, setSession, clearSession } from './api.js';
import Login from './components/Login.jsx';
import Dashboard from './components/Dashboard.jsx';
import Threats from './components/Threats.jsx';
import MitreMatrix from './components/MitreMatrix.jsx';
import ComputerModel from './components/ComputerModel.jsx';
import QuantumLab from './components/QuantumLab.jsx';

const TABS = [
  { id: 'dashboard', title: 'Обзор' },
  { id: 'threats', title: 'Threat Intelligence' },
  { id: 'mitre', title: 'MITRE ATT&CK' },
  { id: 'model', title: 'Модель компьютера (CPU)' },
  { id: 'quantum', title: 'Квантовые модели' },
];

export default function App() {
  const [user, setUser] = useState(getToken() ? getUser() : null);
  const [tab, setTab] = useState('dashboard');

  const onLogout = () => { clearSession(); setUser(null); };
  useEffect(() => {
    window.addEventListener('tiq:logout', onLogout);
    return () => window.removeEventListener('tiq:logout', onLogout);
  }, []);

  if (!user) {
    return (
      <Login
        onLogin={(data) => {
          setSession(data.access_token, { username: data.username, role: data.role });
          setUser({ username: data.username, role: data.role });
        }}
      />
    );
  }

  return (
    <div className="layout">
      <header className="topbar">
        <div className="brand">⚛ TI · ATT&amp;CK · <span>Quantum Security</span></div>
        <nav className="tabs">
          {TABS.map((t) => (
            <button key={t.id}
                    className={`tab ${tab === t.id ? 'active' : ''}`}
                    onClick={() => setTab(t.id)}>{t.title}</button>
          ))}
        </nav>
        <div className="userbox">
          <span className="role-badge">{user.role}</span>
          <span className="username">{user.username}</span>
          <button className="btn ghost" onClick={onLogout}>Выход</button>
        </div>
      </header>

      <main className="content">
        {tab === 'dashboard' && <Dashboard />}
        {tab === 'threats' && <Threats role={user.role} />}
        {tab === 'mitre' && <MitreMatrix />}
        {tab === 'model' && <ComputerModel />}
        {tab === 'quantum' && <QuantumLab />}
      </main>

      <footer className="footer">
        Платформа threat-разведки: MITRE ATT&amp;CK + описание компьютерной модели
        через квантовые модели CPU · FastAPI + React · Docker Compose
      </footer>
    </div>
  );
}
