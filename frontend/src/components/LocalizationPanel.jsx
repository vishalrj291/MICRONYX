import React from 'react';
import { Target, AlertCircle, Loader2, Clock, BarChart3, Cpu } from 'lucide-react';

function formatConfidence(v) {
  if (v == null) return '—';
  return (v * 100).toFixed(4) + '%';
}

function formatCoord(v) {
  if (v == null) return '—';
  return v.toFixed(3) + ' px';
}

function formatRuntime(v) {
  if (v == null) return '—';
  return v.toFixed(1) + ' ms';
}

export default function LocalizationPanel({ result, analysisState, error }) {
  const hasResult = result && result.predicted_x != null;
  const conf = hasResult ? result.confidence : null;
  const confPct = conf != null ? conf * 100 : 0;

  return (
    <div
      style={{
        background: 'var(--bg-panel)',
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      {/* Panel header */}
      <div className="panel-header">
        <Target size={10} strokeWidth={1.5} />
        Localization
      </div>

      {/* Status badge */}
      <div className="panel-section">
        <div
          className={`detection-badge ${
            analysisState === 'success'
              ? 'detected'
              : analysisState === 'loading'
              ? 'running'
              : analysisState === 'error'
              ? 'error'
              : 'waiting'
          }`}
        >
          {analysisState === 'loading' && <Loader2 size={8} className="spin" />}
          {analysisState === 'success' && (
            <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--status-ok)', display: 'inline-block' }} />
          )}
          {analysisState === 'error' && <AlertCircle size={8} />}

          <span>
            {analysisState === 'success'
              ? 'Target Detected'
              : analysisState === 'loading'
              ? 'Running…'
              : analysisState === 'error'
              ? 'Localization Failed'
              : 'Awaiting Acquisition'}
          </span>
        </div>
      </div>

      {/* Coordinates */}
      <div className="panel-section">
        <div className="section-label">Predicted Center</div>

        <div style={{ display: 'flex', gap: '6px', marginBottom: '2px' }}>
          {/* X */}
          <div className="result-coord-block" style={{ flex: 1 }}>
            <span className="result-coord-axis">X</span>
            <span className="result-coord-value" style={{ fontSize: '15px' }}>
              {hasResult ? result.predicted_x.toFixed(3) : '—'}
            </span>
            <span className="result-coord-unit">px</span>
          </div>
          {/* Y */}
          <div className="result-coord-block" style={{ flex: 1 }}>
            <span className="result-coord-axis">Y</span>
            <span className="result-coord-value" style={{ fontSize: '15px' }}>
              {hasResult ? result.predicted_y.toFixed(3) : '—'}
            </span>
            <span className="result-coord-unit">px</span>
          </div>
        </div>
      </div>

      {/* Confidence */}
      <div className="panel-section">
        <div className="section-label">Model Confidence</div>
        <div
          style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: '20px',
            fontWeight: 600,
            color: hasResult ? 'var(--accent)' : 'var(--text-dim)',
            lineHeight: 1.1,
          }}
        >
          {formatConfidence(conf)}
        </div>
        <div className="confidence-bar-track">
          <div
            className="confidence-bar-fill"
            style={{ width: `${Math.min(confPct, 100)}%` }}
          />
        </div>
      </div>

      {/* Metrics */}
      <div className="panel-section">
        <div className="section-label">Metrics</div>

        <div className="data-row">
          <span className="data-label">
            <BarChart3 size={9} style={{ display: 'inline', marginRight: 3, verticalAlign: 'middle' }} strokeWidth={1.5} />
            Candidates
          </span>
          <span className="data-value mono">
            {hasResult ? result.candidates : '—'}
          </span>
        </div>

        <div className="data-row">
          <span className="data-label">
            <Clock size={9} style={{ display: 'inline', marginRight: 3, verticalAlign: 'middle' }} strokeWidth={1.5} />
            Runtime
          </span>
          <span className="data-value mono accent">
            {formatRuntime(hasResult ? result.runtime_ms : null)}
          </span>
        </div>

        <div className="data-row">
          <span className="data-label">API Wall Time</span>
          <span className="data-value mono">
            {formatRuntime(hasResult ? result.api_wall_time_ms : null)}
          </span>
        </div>
      </div>

      {/* Engine info */}
      <div className="panel-section">
        <div className="section-label">Engine</div>

        <div className="data-row">
          <span className="data-label">
            <Cpu size={9} style={{ display: 'inline', marginRight: 3, verticalAlign: 'middle' }} strokeWidth={1.5} />
            Model
          </span>
          <span className="data-value mono" style={{ fontSize: '9px' }}>
            XGBoost Ranker
          </span>
        </div>

        <div className="data-row">
          <span className="data-label">Candidate Gen.</span>
          <span className="data-value mono" style={{ fontSize: '9px' }}>DoG</span>
        </div>

        <div className="data-row">
          <span className="data-label">Acquisition</span>
          <span className="data-value mono" style={{ fontSize: '9px' }}>Canonical</span>
        </div>
      </div>

      {/* Error display */}
      {analysisState === 'error' && error && (
        <div
          className="panel-section"
          style={{
            borderTop: '1px solid rgba(231,76,60,0.25)',
            marginTop: 'auto',
          }}
        >
          <div
            style={{
              fontSize: '9px',
              color: 'var(--status-error)',
              fontFamily: "'JetBrains Mono', monospace",
              lineHeight: 1.5,
              wordBreak: 'break-word',
            }}
          >
            {error}
          </div>
        </div>
      )}
    </div>
  );
}
