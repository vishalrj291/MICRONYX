import React from 'react';
import { List } from 'lucide-react';

function formatProb(v) {
  return (v * 100).toFixed(4) + '%';
}

function calcDistance(x, y, refX, refY) {
  if (refX == null || refY == null) return null;
  return Math.sqrt((x - refX) ** 2 + (y - refY) ** 2).toFixed(1);
}

export default function CandidateTable({ result, selectedCandidate, onSelectCandidate }) {
  const candidates = result?.top_candidates ?? [];
  const topX = result?.predicted_x;
  const topY = result?.predicted_y;
  const maxProb = candidates[0]?.probability ?? 1;

  if (!result) {
    return (
      <div
        style={{
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          background: 'var(--bg-panel)',
        }}
      >
        <div className="panel-header">
          <List size={10} strokeWidth={1.5} />
          Candidate Ranking
        </div>
        <div className="empty-state" style={{ padding: '20px' }}>
          <span className="empty-state-label">No candidates</span>
          <span className="empty-state-sub">Run analysis to populate candidate table</span>
        </div>
      </div>
    );
  }

  return (
    <div
      style={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        background: 'var(--bg-panel)',
        overflow: 'hidden',
      }}
    >
      <div className="panel-header" style={{ justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <List size={10} strokeWidth={1.5} />
          Candidate Ranking
        </div>
        <span style={{ color: 'var(--text-muted)', fontSize: '8.5px' }}>
          Total candidates: {result.candidates}
        </span>
      </div>

      <div style={{ flex: 1, overflowX: 'auto', overflowY: 'auto' }}>
        <table className="candidate-table">
          <thead>
            <tr>
              <th style={{ width: '32px' }}>Rank</th>
              <th>X (px)</th>
              <th>Y (px)</th>
              <th>Probability</th>
              <th>DoG Score</th>
              <th>Δ Distance</th>
              <th style={{ width: '100px' }}>Confidence Bar</th>
            </tr>
          </thead>
          <tbody>
            {candidates.map((c) => {
              const isSelected =
                selectedCandidate &&
                selectedCandidate.x === c.x &&
                selectedCandidate.y === c.y;
              const dist = calcDistance(c.x, c.y, topX, topY);
              const barWidth = maxProb > 0 ? (c.probability / maxProb) * 100 : 0;

              return (
                <tr
                  key={c.rank}
                  className={isSelected ? 'selected' : ''}
                  onClick={() => onSelectCandidate(isSelected ? null : { x: c.x, y: c.y })}
                >
                  <td className="rank-cell">
                    {String(c.rank).padStart(2, '0')}
                  </td>
                  <td style={{ color: 'var(--cyan)' }}>
                    {c.x.toFixed(1)}
                  </td>
                  <td style={{ color: 'var(--cyan)' }}>
                    {c.y.toFixed(1)}
                  </td>
                  <td>
                    <span style={{ color: c.rank === 1 ? 'var(--accent)' : 'var(--text-secondary)' }}>
                      {formatProb(c.probability)}
                    </span>
                  </td>
                  <td>
                    {c.dog_score.toFixed(6)}
                  </td>
                  <td>
                    {c.rank === 1 ? (
                      <span style={{ color: 'var(--text-dim)', fontSize: '9px' }}>—</span>
                    ) : (
                      <span>{dist} px</span>
                    )}
                  </td>
                  <td>
                    <div className="prob-bar">
                      <div
                        className="prob-fill"
                        style={{ width: `${barWidth}%` }}
                      />
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
