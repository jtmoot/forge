"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

const GRAFANA_URL = process.env.NEXT_PUBLIC_GRAFANA_URL;
const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface NavItem {
  href: string;
  label: string;
  icon: string;
  external?: boolean;
}

interface NavGroup {
  label: string;
  items: NavItem[];
}

const navGroups: NavGroup[] = [
  {
    label: "",
    items: [
      { href: "/", label: "Dashboard", icon: "M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-4 0h4" },
    ],
  },
  {
    label: "Data",
    items: [
      { href: "/projects", label: "Projects", icon: "M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" },
      { href: "/experiments", label: "Experiments", icon: "M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" },
      { href: "/features", label: "Features", icon: "M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4" },
    ],
  },
  {
    label: "Models",
    items: [
      { href: "/models", label: "Registry", icon: "M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" },
      { href: "/drift", label: "Drift", icon: "M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" },
      { href: "/serving", label: "Serving", icon: "M13 10V3L4 14h7v7l9-11h-7z" },
    ],
  },
  {
    label: "Tools",
    items: [
      { href: "/agent", label: "Agent", icon: "M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" },
      ...(GRAFANA_URL
        ? [{ href: GRAFANA_URL, label: "Monitoring", icon: "M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z", external: true }]
        : []),
    ],
  },
];

export default function Sidebar() {
  const pathname = usePathname();
  const [apiUp, setApiUp] = useState<boolean | null>(null);

  useEffect(() => {
    fetch(`${API_URL}/health`, { method: "GET" })
      .then((r) => setApiUp(r.ok))
      .catch(() => setApiUp(false));
  }, []);

  function isActive(href: string): boolean {
    if (href === "/") return pathname === "/";
    return pathname.startsWith(href);
  }

  return (
    <aside className="w-64 bg-forge-bg-raised border-r border-forge-border flex flex-col fixed top-0 left-0 h-full z-30">
      {/* Logo */}
      <div className="px-5 py-5 border-b border-forge-border">
        <h1 className="text-sm font-semibold text-forge-text tracking-tight">forge</h1>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-3 pt-3 pb-3 overflow-y-auto">
        {navGroups.map((group) => (
          <div key={group.label || "top"} className={group.label ? "mt-5" : ""}>
            {group.label && (
              <p className="text-2xs font-medium uppercase tracking-widest text-forge-muted px-3 mb-1.5">
                {group.label}
              </p>
            )}
            <div className="space-y-0.5">
              {group.items.map((item) => {
                const active = !item.external && isActive(item.href);
                const cls = [
                  "flex items-center gap-2.5 px-3 py-2 rounded-md text-sm transition-colors relative focus-visible:outline focus-visible:outline-2 focus-visible:outline-forge-accent focus-visible:outline-offset-[-2px]",
                  active
                    ? "bg-white/[0.06] text-forge-text font-medium"
                    : "text-forge-secondary hover:text-forge-text hover:bg-white/[0.03]",
                ].join(" ");

                const icon = (
                  <svg className="w-[18px] h-[18px] flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d={item.icon} />
                  </svg>
                );

                if (item.external) {
                  return (
                    <a key={item.href} href={item.href} target="_blank" rel="noopener noreferrer" className={cls}>
                      {icon}
                      {item.label}
                      <svg className="w-3 h-3 ml-auto text-forge-muted" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                      </svg>
                    </a>
                  );
                }

                return (
                  <Link key={item.href} href={item.href} className={cls}>
                    {active && (
                      <span className="absolute left-0 top-1/2 -translate-y-1/2 w-[3px] h-4 rounded-r bg-forge-accent" />
                    )}
                    {icon}
                    {item.label}
                  </Link>
                );
              })}
            </div>
          </div>
        ))}
      </nav>

      {/* Bottom status */}
      <div className="px-5 py-4 border-t border-forge-border">
        <div className="flex items-center gap-2 text-2xs text-forge-muted">
          <span
            className={`w-1.5 h-1.5 rounded-full ${
              apiUp === true ? "bg-forge-success" : apiUp === false ? "bg-forge-error" : "bg-forge-muted"
            }`}
          />
          {apiUp === true ? "API connected" : apiUp === false ? "API unreachable" : "Checking..."}
        </div>
      </div>
    </aside>
  );
}
