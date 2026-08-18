import React from 'react';
import { Cpu, Activity, CheckCircle2, AlertCircle, Loader2 } from 'lucide-react';

export default function Header({ health, healthLoading }) {
  const isOnline = health?.status === 'healthy';
  const localizerOk = health?.localizer_available;
  const modelOk = health?.model_available;
  const allOk = isOnline && localizerOk && modelOk;

  return (
    <header
      style={{
        gridColumn: '1 / -1',
        background: 'var(--bg-panel)',
        borderBottom: '1px solid var(--border-dim)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0 16px',
        userSelect: 'none',
      }}
    >
      {/* LEFT — Brand */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
        <Cpu size={14} color="var(--accent)" strokeWidth={1.5} />
        <div style={{ display: 'flex', alignItems: 'baseline', gap: '6px' }}>
          <span
            style={{
              fontFamily: "'JetBrains Mono', monospace",
              fontWeight: 600,
              fontSize: '13px',
              letterSpacing: '0.08em',
              color: 'var(--text-primary)',
            }}
          >
            MICRONYX
          </span>
          <span
            style={{
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: '10px',
              color: 'var(--text-dim)',
              letterSpacing: '0.04em',
            }}
          >
            PS02
          </span>
          <span
            style={{
              fontSize: '9px',
              color: 'var(--text-muted)',
              letterSpacing: '0.06em',
            }}
          >
            •
          </span>
          <span
            style={{
              fontSize: '9px',
              color: 'var(--text-dim)',
              letterSpacing: '0.08em',
              textTransform: 'uppercase',
            }}
          >
            Physics-Aware Target Localization
          </span>
        </div>
      </div>

      {/* RIGHT — Status indicators */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        {/* API status badge */}
        <div className="header-badge" style={isOnline ? { borderColor: 'rgba(46,204,113,0.2)', color: 'rgba(46,204,113,0.6)' } : {}}>
          {healthLoading ? (
            <Loader2 size={7} className="spin" />
          ) : isOnline ? (
            <span style={{ width: 5, height: 5, borderRadius: '50%', background: 'var(--status-ok)', display: 'inline-block' }} />
          ) : (
            <span style={{ width: 5, height: 5, borderRadius: '50%', background: 'var(--status-error)', display: 'inline-block' }} />
          )}
          API {isOnline ? 'Connected' : 'Offline'}
        </div>

        {/* Model badge */}
        <div
          className="header-badge"
          style={
            modelOk
              ? { borderColor: 'rgba(46,204,113,0.2)', color: 'rgba(46,204,113,0.6)' }
              : {}
          }
        >
          <span
            style={{
              width: 5,
              height: 5,
              borderRadius: '50%',
              background: modelOk ? 'var(--status-ok)' : 'var(--status-idle)',
              display: 'inline-block',
            }}
          />
          Model {modelOk ? 'Loaded' : 'Unavailable'}
        </div>

        {/* System ready */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            paddingLeft: '10px',
            borderLeft: '1px solid var(--border-dim)',
          }}
        >
          <span
            className={`status-dot ${allOk ? 'ok pulse' : healthLoading ? 'warn' : 'error'}`}
          />
          <span
            style={{
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: '10px',
              fontWeight: 600,
              letterSpacing: '0.1em',
              textTransform: 'uppercase',
              color: allOk
                ? 'var(--status-ok)'
                : healthLoading
                ? 'var(--accent)'
                : 'var(--status-error)',
            }}
          >
            {healthLoading ? 'Connecting…' : allOk ? 'System Ready' : 'System Offline'}
          </span>
        </div>
      </div>
    </header>
  );
}
