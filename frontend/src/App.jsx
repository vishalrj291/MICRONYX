import React, { useState, useEffect, useCallback, useRef } from 'react';
import Header from './components/Header.jsx';
import AcquisitionPanel from './components/AcquisitionPanel.jsx';
import InspectionView from './components/InspectionView.jsx';
import LocalizationPanel from './components/LocalizationPanel.jsx';
import CandidateTable from './components/CandidateTable.jsx';
import ReferenceView from './components/ReferenceView.jsx';
import { checkHealth, runLocalize, fetchDemoFile } from './services/api.js';

export default function App() {
  // ── Health ──────────────────────────────────────────────────
  const [health, setHealth] = useState(null);
  const [healthLoading, setHealthLoading] = useState(true);

  useEffect(() => {
    let alive = true;

    async function poll() {
      try {
        const h = await checkHealth();
        if (alive) {
          setHealth(h);
          setHealthLoading(false);
        }
      } catch {
        if (alive) {
          setHealth(null);
          setHealthLoading(false);
        }
      }
    }

    poll();
    const id = setInterval(poll, 10_000);
    return () => { alive = false; clearInterval(id); };
  }, []);

  // ── Files ────────────────────────────────────────────────────
  const [searchFile, setSearchFile]   = useState(null);
  const [referenceFile, setReferenceFile] = useState(null);

  const searchUrlRef   = useRef(null);
  const referenceUrlRef = useRef(null);

  const [searchUrl, setSearchUrl]     = useState(null);
  const [referenceUrl, setReferenceUrl] = useState(null);

  function setSearchFileAndUrl(f) {
    if (searchUrlRef.current) URL.revokeObjectURL(searchUrlRef.current);
    setSearchFile(f);
    if (f) {
      const u = URL.createObjectURL(f);
      searchUrlRef.current = u;
      setSearchUrl(u);
    } else {
      searchUrlRef.current = null;
      setSearchUrl(null);
    }
  }

  function setReferenceFileAndUrl(f) {
    if (referenceUrlRef.current) URL.revokeObjectURL(referenceUrlRef.current);
    setReferenceFile(f);
    if (f) {
      const u = URL.createObjectURL(f);
      referenceUrlRef.current = u;
      setReferenceUrl(u);
    } else {
      referenceUrlRef.current = null;
      setReferenceUrl(null);
    }
  }

  // ── Load demo ────────────────────────────────────────────────
  const handleLoadDemo = useCallback(async () => {
    try {
      const [sf, rf] = await Promise.all([
        fetchDemoFile('search'),
        fetchDemoFile('reference'),
      ]);
      setSearchFileAndUrl(sf);
      setReferenceFileAndUrl(rf);
    } catch (err) {
      alert('Failed to load demo images: ' + err.message);
    }
  }, []);

  // ── Analysis state ───────────────────────────────────────────
  const [analysisState, setAnalysisState] = useState('idle');
  // 'idle' | 'loading' | 'success' | 'error'
  const [result, setResult]   = useState(null);
  const [error, setError]     = useState(null);

  const handleAnalyze = useCallback(async () => {
    if (!searchFile || !referenceFile) return;
    setAnalysisState('loading');
    setResult(null);
    setError(null);
    setSelectedCandidate(null);

    try {
      const res = await runLocalize(searchFile, referenceFile);
      setResult(res);
      setAnalysisState('success');
    } catch (err) {
      setError(err.message || 'Unknown error');
      setAnalysisState('error');
    }
  }, [searchFile, referenceFile]);

  // ── Selected candidate (from table click) ────────────────────
  const [selectedCandidate, setSelectedCandidate] = useState(null);

  // Reset selected candidate when new result arrives
  useEffect(() => {
    setSelectedCandidate(null);
  }, [result]);

  // ── Layout ───────────────────────────────────────────────────
  return (
    <div
      style={{
        display: 'grid',
        gridTemplateRows: '42px 1fr 190px 155px',
        gridTemplateColumns: '240px 1fr 220px',
        height: '100vh',
        width: '100vw',
        background: 'var(--bg-void)',
        overflow: 'hidden',
      }}
    >
      {/* ── Header ─────────────────────────────────── row 1, all cols */}
      <div style={{ gridColumn: '1 / -1', gridRow: '1' }}>
        <Header health={health} healthLoading={healthLoading} />
      </div>

      {/* ── Left panel ──────────────────────────────── row 2+3, col 1 */}
      <div
        style={{
          gridColumn: '1',
          gridRow: '2 / 4',
          borderRight: '1px solid var(--border-dim)',
          overflow: 'hidden',
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        <AcquisitionPanel
          searchFile={searchFile}
          referenceFile={referenceFile}
          onSearchFile={setSearchFileAndUrl}
          onReferenceFile={setReferenceFileAndUrl}
          onLoadDemo={handleLoadDemo}
          onAnalyze={handleAnalyze}
          analysisState={analysisState}
        />
      </div>

      {/* ── Center top — Inspection view ────────────── row 2, col 2 */}
      <div
        style={{
          gridColumn: '2',
          gridRow: '2',
          overflow: 'hidden',
          borderBottom: 'none',
        }}
      >
        <InspectionView
          imageUrl={searchUrl}
          result={result}
          selectedCandidate={selectedCandidate}
          analysisState={analysisState}
        />
      </div>

      {/* ── Right panel ─────────────────────────────── row 2+3, col 3 */}
      <div
        style={{
          gridColumn: '3',
          gridRow: '2 / 4',
          borderLeft: '1px solid var(--border-dim)',
          overflow: 'hidden',
        }}
      >
        <LocalizationPanel
          result={result}
          analysisState={analysisState}
          error={error}
        />
      </div>

      {/* ── Center bottom — Reference view ──────────── row 3, col 2 */}
      <div
        style={{
          gridColumn: '2',
          gridRow: '3',
          borderTop: '1px solid var(--border-dim)',
          overflow: 'hidden',
        }}
      >
        <ReferenceView referenceUrl={referenceUrl} result={result} />
      </div>

      {/* ── Candidate table ──────────────────────── row 4, all cols */}
      <div
        style={{
          gridColumn: '1 / -1',
          gridRow: '4',
          borderTop: '1px solid var(--border-dim)',
          overflow: 'hidden',
        }}
      >
        <CandidateTable
          result={result}
          selectedCandidate={selectedCandidate}
          onSelectCandidate={setSelectedCandidate}
        />
      </div>
    </div>
  );
}
