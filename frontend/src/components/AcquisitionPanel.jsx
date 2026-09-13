import React, { useRef } from 'react';
import {
  Upload,
  FolderOpen,
  Play,
  Loader2,
  Image,
  FlaskConical,
} from 'lucide-react';

const ACQUISITION_PARAMS = [
  { label: 'Search Image', value: '1000 × 1000 px' },
  { label: 'Reference Image', value: '1000 × 1000 px' },
  { label: 'Sampling Ratio', value: '10×' },
  { label: 'Search PPU', value: '5.0 px/unit' },
  { label: 'Reference PPU', value: '50.0 px/unit' },
  { label: 'Physical FOV', value: '200 × 200 units' },
];

function UploadZone({ label, file, onFile, accept = '.png,.tiff,.tif' }) {
  const inputRef = useRef(null);

  return (
    <div style={{ marginBottom: '6px' }}>
      <div className="section-label" style={{ marginBottom: '3px' }}>
        {label}
      </div>
      <div
        className={`upload-zone ${file ? 'loaded' : ''}`}
        onClick={() => inputRef.current?.click()}
        title={file ? file.name : `Click to upload ${label}`}
      >
        <Image size={12} color={file ? 'var(--accent)' : 'var(--text-dim)'} strokeWidth={1.5} />
        <span className="upload-filename">
          {file ? file.name : 'No file selected'}
        </span>
        <Upload size={10} color="var(--text-dim)" strokeWidth={1.5} />
        <input
          ref={inputRef}
          type="file"
          accept={accept}
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) onFile(f);
            e.target.value = '';
          }}
        />
      </div>
    </div>
  );
}

export default function AcquisitionPanel({
  searchFile,
  referenceFile,
  onSearchFile,
  onReferenceFile,
  onLoadDemo,
  onAnalyze,
  analysisState,  // 'idle' | 'loading' | 'success' | 'error'
}) {
  const canAnalyze = searchFile && referenceFile && analysisState !== 'loading';
  const isLoading = analysisState === 'loading';

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
        <FolderOpen size={10} strokeWidth={1.5} />
        Acquisition
      </div>

      {/* Acquisition parameters */}
      <div className="panel-section">
        <div className="section-label">Parameters</div>
        {ACQUISITION_PARAMS.map(({ label, value }) => (
          <div className="data-row" key={label}>
            <span className="data-label">{label}</span>
            <span className="data-value mono">{value}</span>
          </div>
        ))}
      </div>

      {/* File inputs */}
      <div className="panel-section" style={{ flex: '0 0 auto' }}>
        <div className="section-label">Image Input</div>

        <UploadZone
          label="Search Image"
          file={searchFile}
          onFile={onSearchFile}
        />

        <UploadZone
          label="Reference Image"
          file={referenceFile}
          onFile={onReferenceFile}
        />

        {/* Load Demo */}
        <button
          className="btn btn-ghost"
          style={{ width: '100%', marginTop: '4px', gap: '5px' }}
          onClick={onLoadDemo}
          disabled={isLoading}
        >
          <FlaskConical size={11} strokeWidth={1.5} />
          Load Demo Data
        </button>
      </div>

      {/* Analyze button */}
      <div className="panel-section" style={{ marginTop: 'auto' }}>
        <button
          id="btn-analyze"
          className={`btn btn-primary ${isLoading ? 'running' : ''}`}
          style={{ width: '100%', padding: '8px', fontSize: '11px', gap: '6px' }}
          disabled={!canAnalyze}
          onClick={onAnalyze}
        >
          {isLoading ? (
            <>
              <Loader2 size={12} className="spin" />
              Running Localization…
            </>
          ) : (
            <>
              <Play size={12} strokeWidth={2} />
              Analyze
            </>
          )}
        </button>

        {!searchFile && !referenceFile && (
          <p
            style={{
              fontSize: '9px',
              color: 'var(--text-muted)',
              textAlign: 'center',
              marginTop: '6px',
              lineHeight: 1.5,
            }}
          >
            Upload images or load demo data to begin acquisition.
          </p>
        )}
      </div>

      {/* Pipeline label at bottom */}
      <div className="panel-section" style={{ marginTop: 0 }}>
        <div className="section-label">Pipeline</div>
        <PipelineMini isRunning={isLoading} />
      </div>
    </div>
  );
}

/* Compact inline pipeline for the left panel */
function PipelineMini({ isRunning }) {
  const steps = [
    'Canonical Acquisition',
    'Multi-Res Observation',
    'DoG Candidate Gen.',
    'Multi-Scale Features',
    'XGBoost Ranking',
    'Target Localization',
  ];

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'stretch',
        gap: 0,
      }}
    >
      {steps.map((step, i) => (
        <div key={step} className="pipeline-step">
          <div
            className={`pipeline-node ${isRunning ? 'active' : ''}`}
            style={
              isRunning
                ? { animationDelay: `${i * 0.15}s` }
                : {}
            }
          >
            {step}
          </div>
          {i < steps.length - 1 && (
            <div className="pipeline-connector" />
          )}
        </div>
      ))}
    </div>
  );
}
