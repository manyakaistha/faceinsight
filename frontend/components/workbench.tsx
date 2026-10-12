"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState, ReactNode } from "react";
import {
  ScanFace,
  LayoutDashboard,
  ScanLine,
  Fingerprint,
  ChartNoAxesCombined,
  Network,
  BookOpen,
  ArrowUpRight,
  Menu,
  X,
  Cpu,
  CircleHelp,
} from "lucide-react";

export const navigation = [
  { href: "/", label: "Overview", icon: LayoutDashboard },
  { href: "/inference", label: "Live inference", icon: ScanLine },
  { href: "/explainability", label: "Explainability", icon: Fingerprint },
  { href: "/performance", label: "Performance", icon: ChartNoAxesCombined },
  { href: "/architecture", label: "Architecture", icon: Network },
  { href: "/about", label: "About the project", icon: BookOpen },
];
export async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`/api/v1/${path}`, options);
  const data = await response.json();
  if (!response.ok)
    throw new Error(
      typeof data.detail === "string"
        ? data.detail
        : "The request could not be completed. Please retry.",
    );
  return data;
}
export function PageHeading({
  eyebrow,
  title,
  description,
  children,
}: {
  eyebrow: string;
  title: ReactNode;
  description: string;
  children?: ReactNode;
}) {
  return (
    <div className="page-heading">
      <div>
        <div className="eyebrow">
          <span />
          {eyebrow}
        </div>
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
      {children}
    </div>
  );
}
export function Workbench({ children }: { children: ReactNode }) {
  const path = usePathname();
  const [open, setOpen] = useState(false);
  const [status, setStatus] = useState<"connecting" | "ready" | "offline">(
    "connecting",
  );
  const [device, setDevice] = useState("—");
  useEffect(() => {
    let alive = true;
    const check = () =>
      api<{ device: string }>("model/info")
        .then((d) => {
          if (alive) {
            setStatus("ready");
            setDevice(d.device);
          }
        })
        .catch(() => {
          if (alive) setStatus("offline");
        });
    check();
    const timer = setInterval(check, 30000);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, []);
  useEffect(() => {
    setOpen(false);
  }, [path]);
  const current = navigation.find((n) => n.href === path)?.label || "Workspace";
  return (
    <div className="app-shell">
      <a className="skip-link" href="#content">
        Skip to content
      </a>
      {open && (
        <button
          className="scrim"
          onClick={() => setOpen(false)}
          aria-label="Close navigation"
        />
      )}
      <aside className={`sidebar ${open ? "open" : ""}`}>
        <Link href="/" className="brand">
          <span className="brand-icon">
            <ScanFace size={25} />
          </span>
          <span>
            faceinsight<span className="brand-dot">.</span>
            <small>VISION RESEARCH STUDIO</small>
          </span>
        </Link>
        <Link href="/inference" className="new-analysis">
          <span>＋</span> New analysis <ArrowUpRight size={15} />
        </Link>
        <div className="nav-label">
          WORKSPACE <span>06</span>
        </div>
        <nav aria-label="Main navigation">
          {navigation.map((n) => (
            <Link
              key={n.href}
              href={n.href}
              className={`nav-item ${path === n.href ? "active" : ""}`}
              aria-current={path === n.href ? "page" : undefined}
            >
              <n.icon size={18} />
              {n.label}
              {path === n.href && <span className="active-dot" />}
            </Link>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="workspace-card">
            <div>
              <i className={`status-dot ${status}`} />
              <strong>Local model workspace</strong>
            </div>
            <p>
              {status === "ready"
                ? "Checkpoint loaded. Ready to explore."
                : status === "connecting"
                  ? "Connecting to the model…"
                  : "Model server unavailable."}
            </p>
            <Link href="/architecture">
              Model & system details <ArrowUpRight size={14} />
            </Link>
          </div>
          <div className="sidebar-foot">
            <span>●</span> A closer look at computer vision
          </div>
        </div>
      </aside>
      <div className="workbench">
        <header className="topbar">
          <div className="breadcrumb">
            <button
              className="menu-button icon-button"
              onClick={() => setOpen(!open)}
              aria-label="Toggle navigation"
              aria-expanded={open}
            >
              {open ? <X size={20} /> : <Menu size={20} />}
            </button>
            <span>Research studio</span>
            <span className="slash">/</span>
            <strong>{current}</strong>
          </div>
          <div className="top-actions">
            <span className={`health-pill ${status}`} role="status">
              <i className={`status-dot ${status}`} />
              {status === "ready"
                ? "Model ready"
                : status === "offline"
                  ? "Disconnected"
                  : "Connecting"}
            </span>
            <span className="device-pill">
              <Cpu size={14} />
              {device.toUpperCase()}
            </span>
            <Link
              href="/about"
              className="avatar"
              aria-label="About FaceInsight"
            >
              f<span>i</span>
            </Link>
          </div>
        </header>
        <main id="content" className="content">
          {children}
        </main>
        <footer className="page-footer">
          <span>
            FaceInsight AI <span className="footer-separator">/</span> A local
            vision workbench
          </span>
          <Link href="/about">
            <CircleHelp size={13} /> Understanding the estimates
          </Link>
        </footer>
      </div>
    </div>
  );
}
