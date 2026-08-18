import React, { useRef, useEffect, useCallback, useState } from 'react';
import { ZoomIn, ZoomOut, Maximize2, RotateCcw, Crosshair } from 'lucide-react';

const MARKER_COLOR = '#e8a838';
const MARKER_SECONDARY = 'rgba(232, 168, 56, 0.3)';
const CROSSHAIR_SIZE = 18;
const BOX_HALF = 22;

/**
 * Draw the target marker (crosshair + bounding box + label) on the canvas
 * at the given image-space coordinates, applying the current transform.
 */
function drawTargetMarker(ctx, ix, iy, tx, ty, scale, label) {
  // Convert image coordinates to canvas coordinates
  const cx = tx + ix * scale;
  const cy = ty + iy * scale;

  ctx.save();

  // Outer glow ring
  ctx.strokeStyle = MARKER_SECONDARY;
  ctx.lineWidth = 1;
  ctx.setLineDash([]);
  ctx.beginPath();
  ctx.arc(cx, cy, BOX_HALF * scale * 0.9, 0, Math.PI * 2);
  ctx.stroke();

  // Bounding box (dashed)
  const bx = BOX_HALF * scale;
  ctx.strokeStyle = MARKER_COLOR;
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 2]);
  ctx.strokeRect(cx - bx, cy - bx, bx * 2, bx * 2);
  ctx.setLineDash([]);

  // Crosshair
  const cs = CROSSHAIR_SIZE * Math.min(scale, 1.5);
  ctx.strokeStyle = MARKER_COLOR;
  ctx.lineWidth = 1;

  // Horizontal
  ctx.beginPath();
  ctx.moveTo(cx - cs, cy);
  ctx.lineTo(cx - 4 * scale, cy);
  ctx.moveTo(cx + 4 * scale, cy);
  ctx.lineTo(cx + cs, cy);
  ctx.stroke();

  // Vertical
  ctx.beginPath();
  ctx.moveTo(cx, cy - cs);
  ctx.lineTo(cx, cy - 4 * scale);
  ctx.moveTo(cx, cy + 4 * scale);
  ctx.lineTo(cx, cy + cs);
  ctx.stroke();

  // Center dot
  ctx.fillStyle = MARKER_COLOR;
  ctx.beginPath();
  ctx.arc(cx, cy, 2, 0, Math.PI * 2);
  ctx.fill();

  // Label
  if (label) {
    const lx = cx + bx + 4;
    const ly = cy - bx;
    ctx.font = "500 9px 'JetBrains Mono', monospace";
    ctx.fillStyle = MARKER_COLOR;
    label.forEach((line, i) => {
      ctx.fillText(line, lx, ly + i * 12);
    });
  }

  ctx.restore();
}

/**
 * Draw a subtle grid over the image.
 */
function drawGrid(ctx, tx, ty, imgW, imgH, scale, canvasW, canvasH) {
  const gridSpacing = 100 * scale;  // 100px image units
  if (gridSpacing < 20) return;

  ctx.save();
  ctx.strokeStyle = 'rgba(255,255,255,0.04)';
  ctx.lineWidth = 0.5;

  // Vertical lines
  const startX = tx % gridSpacing;
  for (let x = startX; x < canvasW; x += gridSpacing) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, canvasH);
    ctx.stroke();
  }
  // Horizontal lines
  const startY = ty % gridSpacing;
  for (let y = startY; y < canvasH; y += gridSpacing) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(canvasW, y);
    ctx.stroke();
  }

  ctx.restore();
}

export default function InspectionView({
  imageUrl,     // object URL or null
  result,       // LocalizationResult | null
  selectedCandidate,  // { x, y } | null — from clicking a candidate row
  analysisState,
}) {
  const containerRef = useRef(null);
  const canvasRef = useRef(null);
  const imgRef = useRef(null);

  // Pan/zoom state
  const stateRef = useRef({ tx: 0, ty: 0, scale: 1, isDragging: false, lastX: 0, lastY: 0 });
  const [cursorPos, setCursorPos] = useState(null);   // { ix, iy } image coords
  const [fitted, setFitted] = useState(false);

  // Load image
  useEffect(() => {
    if (!imageUrl) {
      imgRef.current = null;
      setFitted(false);
      redraw();
      return;
    }
    const img = new window.Image();
    img.onload = () => {
      imgRef.current = img;
      fitToView();
    };
    img.src = imageUrl;
  }, [imageUrl]);

  const fitToView = useCallback(() => {
    const container = containerRef.current;
    const img = imgRef.current;
    if (!container || !img) return;

    const { width: cw, height: ch } = container.getBoundingClientRect();
    const scaleX = cw / img.naturalWidth;
    const scaleY = ch / img.naturalHeight;
    const scale = Math.min(scaleX, scaleY) * 0.95;
    const tx = (cw - img.naturalWidth * scale) / 2;
    const ty = (ch - img.naturalHeight * scale) / 2;

    stateRef.current = { ...stateRef.current, tx, ty, scale };
    setFitted(true);
    redraw();
  }, []);

  const redraw = useCallback(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const { width: cw, height: ch } = container.getBoundingClientRect();
    if (canvas.width !== cw || canvas.height !== ch) {
      canvas.width = cw;
      canvas.height = ch;
    }

    const ctx = canvas.getContext('2d');
    const { tx, ty, scale } = stateRef.current;

    ctx.clearRect(0, 0, cw, ch);

    // Image
    const img = imgRef.current;
    if (img) {
      ctx.save();
      ctx.imageSmoothingEnabled = scale < 1;
      ctx.imageSmoothingQuality = 'medium';
      ctx.drawImage(img, tx, ty, img.naturalWidth * scale, img.naturalHeight * scale);
      ctx.restore();

      // Grid overlay
      drawGrid(ctx, tx, ty, img.naturalWidth, img.naturalHeight, scale, cw, ch);

      // Target marker
      const markerSource = selectedCandidate || (result && result.predicted_x != null
        ? { x: result.predicted_x, y: result.predicted_y }
        : null);

      if (markerSource) {
        const label = [
          'TARGET',
          `X ${markerSource.x.toFixed(1)}`,
          `Y ${markerSource.y.toFixed(1)}`,
        ];
        drawTargetMarker(ctx, markerSource.x, markerSource.y, tx, ty, scale, label);
      }
    } else {
      // Empty state text
      ctx.font = "600 10px 'Inter', sans-serif";
      ctx.fillStyle = 'rgba(77,95,114,0.6)';
      ctx.textAlign = 'center';
      ctx.fillText('WAITING FOR ACQUISITION', cw / 2, ch / 2 - 10);
      ctx.font = "400 9px 'Inter', sans-serif";
      ctx.fillStyle = 'rgba(46,61,78,0.8)';
      ctx.fillText('Select images to begin analysis', cw / 2, ch / 2 + 6);
    }
  }, [result, selectedCandidate]);

  // Redraw on result/candidate change
  useEffect(() => { redraw(); }, [result, selectedCandidate, redraw]);

  // Fit on mount / window resize
  useEffect(() => {
    fitToView();
    const observer = new ResizeObserver(() => {
      if (imgRef.current) fitToView();
      else redraw();
    });
    if (containerRef.current) observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, [fitToView]);

  // Mouse pan
  const onMouseDown = useCallback((e) => {
    stateRef.current.isDragging = true;
    stateRef.current.lastX = e.clientX;
    stateRef.current.lastY = e.clientY;
  }, []);

  const onMouseMove = useCallback((e) => {
    const { isDragging, lastX, lastY, tx, ty, scale } = stateRef.current;

    // Update cursor image coordinate
    const rect = canvasRef.current?.getBoundingClientRect();
    if (rect) {
      const ix = (e.clientX - rect.left - tx) / scale;
      const iy = (e.clientY - rect.top - ty) / scale;
      if (ix >= 0 && iy >= 0 && ix <= 1000 && iy <= 1000) {
        setCursorPos({ ix: Math.round(ix), iy: Math.round(iy) });
      } else {
        setCursorPos(null);
      }
    }

    if (!isDragging) return;
    const dx = e.clientX - lastX;
    const dy = e.clientY - lastY;
    stateRef.current.tx = tx + dx;
    stateRef.current.ty = ty + dy;
    stateRef.current.lastX = e.clientX;
    stateRef.current.lastY = e.clientY;
    redraw();
  }, [redraw]);

  const onMouseUp = useCallback(() => {
    stateRef.current.isDragging = false;
  }, []);

  const onMouseLeave = useCallback(() => {
    stateRef.current.isDragging = false;
    setCursorPos(null);
  }, []);

  // Scroll to zoom
  const onWheel = useCallback((e) => {
    e.preventDefault();
    const { tx, ty, scale } = stateRef.current;
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;

    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const delta = e.deltaY > 0 ? 0.85 : 1.18;
    const newScale = Math.max(0.1, Math.min(20, scale * delta));
    const newTx = mx - (mx - tx) * (newScale / scale);
    const newTy = my - (my - ty) * (newScale / scale);

    stateRef.current.tx = newTx;
    stateRef.current.ty = newTy;
    stateRef.current.scale = newScale;
    redraw();
  }, [redraw]);

  // Zoom buttons
  const zoomIn = () => {
    stateRef.current.scale = Math.min(20, stateRef.current.scale * 1.3);
    redraw();
  };
  const zoomOut = () => {
    stateRef.current.scale = Math.max(0.1, stateRef.current.scale / 1.3);
    redraw();
  };
  const resetView = () => {
    fitToView();
  };

  return (
    <div
      style={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        background: 'var(--bg-void)',
      }}
    >
      {/* Toolbar */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '4px 8px',
          background: 'var(--bg-panel)',
          borderBottom: '1px solid var(--border-dim)',
          gap: '6px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <span
            style={{
              fontSize: '9px',
              fontWeight: 600,
              letterSpacing: '0.12em',
              textTransform: 'uppercase',
              color: 'var(--text-dim)',
            }}
          >
            Search Image
          </span>
          <span style={{ color: 'var(--border-bright)', fontSize: '9px' }}>•</span>
          <span
            style={{
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: '9px',
              color: 'var(--text-muted)',
            }}
          >
            1000 × 1000 px @ 5.0 px/unit
          </span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '2px' }}>
          {cursorPos && (
            <span
              style={{
                fontFamily: "'JetBrains Mono', monospace",
                fontSize: '9px',
                color: 'var(--cyan)',
                marginRight: '6px',
              }}
            >
              X {cursorPos.ix}  Y {cursorPos.iy}
            </span>
          )}
          <button className="btn btn-ghost" style={{ padding: '2px 5px' }} onClick={zoomIn} title="Zoom In">
            <ZoomIn size={11} strokeWidth={1.5} />
          </button>
          <button className="btn btn-ghost" style={{ padding: '2px 5px' }} onClick={zoomOut} title="Zoom Out">
            <ZoomOut size={11} strokeWidth={1.5} />
          </button>
          <button className="btn btn-ghost" style={{ padding: '2px 5px' }} onClick={resetView} title="Fit to View">
            <Maximize2 size={11} strokeWidth={1.5} />
          </button>
          <button className="btn btn-ghost" style={{ padding: '2px 5px' }} onClick={resetView} title="Reset">
            <RotateCcw size={11} strokeWidth={1.5} />
          </button>
        </div>
      </div>

      {/* Canvas container */}
      <div
        ref={containerRef}
        className="inspection-canvas-container"
        style={{ flex: 1 }}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={onMouseLeave}
        onWheel={onWheel}
      >
        <canvas ref={canvasRef} style={{ display: 'block' }} />

        {/* Loading overlay */}
        {analysisState === 'loading' && (
          <div
            style={{
              position: 'absolute',
              inset: 0,
              background: 'rgba(10,11,13,0.75)',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '10px',
              zIndex: 10,
            }}
          >
            <div
              style={{
                width: 28,
                height: 28,
                border: '2px solid var(--border-mid)',
                borderTopColor: 'var(--accent)',
                borderRadius: '50%',
                animation: 'spin 1s linear infinite',
              }}
            />
            <span
              style={{
                fontFamily: "'JetBrains Mono', monospace",
                fontSize: '10px',
                color: 'var(--accent)',
                letterSpacing: '0.1em',
                textTransform: 'uppercase',
              }}
            >
              Running Localization
            </span>
            <span
              style={{
                fontSize: '9px',
                color: 'var(--text-dim)',
              }}
            >
              DoG candidate generation → feature extraction → XGBoost ranking
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
