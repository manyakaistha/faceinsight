"use client";
import { useEffect, useRef } from "react";

// Seed geometry once. The canvas contains no text or interactive controls.
export default function Field({ compact = false }: { compact?: boolean }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current!;
    const ctx = canvas.getContext("2d")!;
    let seed = 4817;
    const random = () => {
      seed |= 0;
      seed = (seed + 0x6d2b79f5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    const points = Array.from({ length: 92 }, (_, i) => {
      const angle = (i / 92) * Math.PI * 2;
      const radius = 0.8 + random() * 0.2;
      return {
        x: Math.cos(angle) * 0.29 * radius,
        y: Math.sin(angle) * 0.41 * radius,
        phase: random() * 6.28,
      };
    });
    const inner = [
      { x: -0.11, y: -0.1, phase: 1 },
      { x: 0.11, y: -0.1, phase: 2 },
      { x: 0, y: 0.06, phase: 3 },
      { x: -0.09, y: 0.22, phase: 4 },
      { x: 0.09, y: 0.22, phase: 5 },
    ];
    let w = 0,
      h = 0,
      frame = 0,
      last = 0,
      time = 0,
      visible = true,
      px = 0,
      py = 0,
      tx = 0,
      ty = 0;
    const motion = window.matchMedia("(prefers-reduced-motion: reduce)");
    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      w = rect.width;
      h = rect.height;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      draw(time);
    };
    const draw = (t: number) => {
      ctx.clearRect(0, 0, w, h);
      const cx = w / 2 + px * 10,
        cy = h / 2 + py * 10,
        scale = Math.min(w, h);
      ctx.strokeStyle = "rgba(41,107,109,.13)";
      ctx.lineWidth = 1;
      for (let x = 20; x < w; x += 24) {
        for (let y = 20; y < h; y += 24) {
          ctx.fillStyle = "rgba(41,107,109,.13)";
          ctx.fillRect(x, y, 1, 1);
        }
      }
      const all = [...points, ...inner];
      const pos = (p: (typeof points)[number]) => ({
        x: cx + p.x * scale + Math.sin(t * 0.0004 + p.phase) * 2,
        y: cy + p.y * scale + Math.cos(t * 0.0004 + p.phase) * 2,
      });
      points.forEach((p, i) => {
        const a = pos(p),
          b = pos(points[(i + 1) % points.length]);
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
      });
      inner.forEach((p, i) => {
        const a = pos(p);
        for (let j = i + 1; j < inner.length; j++) {
          const b = pos(inner[j]);
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }
        for (let j = i * 18; j < i * 18 + 14; j += 3) {
          const b = pos(points[j % 92]);
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }
      });
      all.forEach((p, i) => {
        const a = pos(p);
        ctx.beginPath();
        ctx.arc(a.x, a.y, i >= 92 ? 4 : 1.7, 0, Math.PI * 2);
        ctx.fillStyle = i >= 92 ? "#b86f4c" : "#44898a";
        ctx.fill();
      });
      const scan = cy + Math.sin(t * 0.00065) * 0.35 * scale;
      ctx.strokeStyle = "rgba(184,111,76,.28)";
      ctx.beginPath();
      ctx.moveTo(cx - scale * 0.33, scan);
      ctx.lineTo(cx + scale * 0.33, scan);
      ctx.stroke();
      const side = scale * 0.72;
      ctx.strokeStyle = "rgba(41,107,109,.45)";
      for (const [x, y, sx, sy] of [
        [cx - side / 2, cy - side / 2, 1, 1],
        [cx + side / 2, cy - side / 2, -1, 1],
        [cx - side / 2, cy + side / 2, 1, -1],
        [cx + side / 2, cy + side / 2, -1, -1],
      ]) {
        ctx.beginPath();
        ctx.moveTo(x + sx * 16, y);
        ctx.lineTo(x, y);
        ctx.lineTo(x, y + sy * 16);
        ctx.stroke();
      }
    };
    const tick = (now: number) => {
      const dt = Math.min(64, last ? now - last : 16);
      last = now;
      time += dt;
      const ease = 1 - Math.exp(-dt / 180);
      px += (tx - px) * ease;
      py += (ty - py) * ease;
      draw(time);
      frame = requestAnimationFrame(tick);
    };
    const sync = () => {
      cancelAnimationFrame(frame);
      last = 0;
      if (visible && !document.hidden && !motion.matches)
        frame = requestAnimationFrame(tick);
      else draw(0);
    };
    const move = (e: PointerEvent) => {
      const r = canvas.getBoundingClientRect();
      tx = (e.clientX - r.left) / r.width - 0.5;
      ty = (e.clientY - r.top) / r.height - 0.5;
    };
    const leave = () => {
      tx = ty = 0;
    };
    const ro = new ResizeObserver(resize);
    ro.observe(canvas);
    const io = new IntersectionObserver((entries) => {
      visible = entries[0].isIntersecting;
      sync();
    });
    io.observe(canvas);
    document.addEventListener("visibilitychange", sync);
    motion.addEventListener("change", sync);
    canvas.addEventListener("pointermove", move);
    canvas.addEventListener("pointerleave", leave);
    resize();
    sync();
    return () => {
      cancelAnimationFrame(frame);
      ro.disconnect();
      io.disconnect();
      document.removeEventListener("visibilitychange", sync);
      motion.removeEventListener("change", sync);
      canvas.removeEventListener("pointermove", move);
      canvas.removeEventListener("pointerleave", leave);
    };
  }, []);
  return (
    <div className={`field ${compact ? "compact" : ""}`}>
      <canvas ref={ref} aria-hidden="true" />
      <div className="field-caption">
        <span>FACIAL FEATURE SPACE</span>
        <span>224 × 224</span>
      </div>
    </div>
  );
}
