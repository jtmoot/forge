"use client";

import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";

/** Markdown elements styled with Forge's tokens (no typography plugin). */
const components: Components = {
  p: ({ children }) => <p className="mb-3 last:mb-0 leading-relaxed">{children}</p>,
  strong: ({ children }) => <strong className="font-semibold text-forge-text">{children}</strong>,
  em: ({ children }) => <em className="italic">{children}</em>,
  ul: ({ children }) => <ul className="mb-3 last:mb-0 list-disc pl-5 space-y-1">{children}</ul>,
  ol: ({ children }) => <ol className="mb-3 last:mb-0 list-decimal pl-5 space-y-1">{children}</ol>,
  li: ({ children }) => <li className="leading-relaxed marker:text-forge-muted">{children}</li>,
  h1: ({ children }) => <h3 className="mb-2 mt-1 text-sm font-semibold text-forge-text">{children}</h3>,
  h2: ({ children }) => <h3 className="mb-2 mt-1 text-sm font-semibold text-forge-text">{children}</h3>,
  h3: ({ children }) => <h3 className="mb-2 mt-1 text-sm font-semibold text-forge-text">{children}</h3>,
  code: ({ children }) => (
    <code className="font-mono text-xs bg-forge-bg-raised border border-forge-border rounded px-1 py-0.5">{children}</code>
  ),
  pre: ({ children }) => (
    <pre className="mb-3 last:mb-0 overflow-x-auto bg-forge-bg-raised border border-forge-border rounded-md p-3 text-xs">{children}</pre>
  ),
  a: ({ href, children }) => (
    <a href={href} target="_blank" rel="noopener noreferrer" className="text-forge-accent hover:underline">
      {children}
    </a>
  ),
  table: ({ children }) => (
    <div className="mb-3 last:mb-0 overflow-x-auto rounded-md border border-forge-border">
      <table className="w-full text-xs">{children}</table>
    </div>
  ),
  thead: ({ children }) => <thead className="bg-forge-bg-raised">{children}</thead>,
  th: ({ children }) => (
    <th className="px-3 py-2 text-left text-2xs font-medium uppercase tracking-widest text-forge-muted">{children}</th>
  ),
  tr: ({ children }) => <tr className="border-t border-forge-border first:border-t-0">{children}</tr>,
  td: ({ children }) => <td className="px-3 py-2 font-mono tabular-nums">{children}</td>,
};

export default function AgentMarkdown({ content }: { content: string }) {
  return (
    <div className="text-sm">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
        {content}
      </ReactMarkdown>
    </div>
  );
}
