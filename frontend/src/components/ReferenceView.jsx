import React, { useRef, useEffect } from 'react';
import { BookOpen } from 'lucide-react';

export default function ReferenceView({ referenceUrl, result }) {
  const canvasRef = useRef(null);
  const imgRef = useRef(null);

  useEffect(() => {
    if (!referenceUrl) {
      imgRef.current = null;
      draw();
      return;
    }
    const img = new window.Image();
    img.onload = () => {
      imgRef.current = img;
      draw();
    };
    img.src = referenceUrl;
  }, [referenceUrl]);

  useEffect(() => {
    draw();
  }, [result]);

  function draw() {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const W = canvas.clientWidth;
    const H = canvas.clientHeight;
    canvas.width = W;
    canvas.height = H;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, W, H);

    const img = imgRef.current;
    if (!img) {
      ctx.font = "600 9px 'Inter', sans-serif";
      ctx.fillStyle = 'rgba(77,95,114,0.4)';
      ctx.textAlign = 'center';
      ctx.fillText('REFERENCE OBSERVATION', W / 2, H / 2);
      return;
    }

    // Fit image
    const scale = Math.min(W / img.naturalWidth, H / img.naturalHeight) * 0.92;
    const tx = (W - img.naturalWidth * scale) / 2;
    const ty = (H - img.naturalHeight * scale) / 2;

    ctx.drawImage(img, tx, ty, img.naturalWidth * scale, img.naturalHeight * scale);

    // Draw reference center marker (center of the reference image)
    const cx = tx + (img.naturalWidth / 2) * scale;
    const cy = ty + (img.naturalHeight / 2) * scale;

    ctx.strokeStyle = 'rgba(232,168,56,0.6)';
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 2]);
    ctx.strokeRect(cx - 15 * scale, cy - 15 * scale, 30 * scale, 30 * scale);
    ctx.setLineDash([]);

    const cs = 10 * scale;
    ctx.beginPath();
    ctx.moveTo(cx - cs, cy); ctx.lineTo(cx - 3 * scale, cy);
    ctx.moveTo(cx + 3 * scale, cy); ctx.lineTo(cx + cs, cy);
    ctx.moveTo(cx, cy - cs); ctx.lineTo(cx, cy - 3 * scale);
    ctx.moveTo(cx, cy + 3 * scale); ctx.lineTo(cx, cy + cs);
    ctx.strokeStyle = 'rgba(232,168,56,0.7)';
    ctx.stroke();
  }

  return (
    <div
      style={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        background: 'var(--bg-panel)',
        borderTop: 'none',
      }}
    >
      <div className="panel-header" style={{ justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <BookOpen size={10} strokeWidth={1.5} />
          Reference Observation
        </div>
        <span
          style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: '8.5px',
            color: 'var(--text-muted)',
          }}
        >
          1000 × 1000 px · 50 px/unit
        </span>
      </div>

      <div style={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
        <canvas
          ref={canvasRef}
          style={{ display: 'block', width: '100%', height: '100%' }}
        />
      </div>
    </div>
  );
}
