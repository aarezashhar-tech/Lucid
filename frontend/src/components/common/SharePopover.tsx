import React, { useState } from "react";
import { MatIcon } from "./MatIcon";

export function SharePopover({
  isOpen,
  onClose,
  title,
  verdict,
}: {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  verdict: string;
}) {
  const [copied, setCopied] = useState(false);
  const currentUrl = typeof window !== "undefined" ? window.location.href : "https://lucid-factcheck.local";
  const shareText = `Fact Check: "${title}" is rated ${verdict} on Lucid AI Fact Verification Platform.`;

  if (!isOpen) return null;

  const handleCopy = () => {
    navigator.clipboard?.writeText(currentUrl);
    setCopied(true);
    setTimeout(() => {
      setCopied(false);
      onClose();
    }, 1800);
  };

  const shareOptions = [
    {
      name: "X (Twitter)",
      icon: (
        <svg className="w-4 h-4 fill-current" viewBox="0 0 24 24">
          <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 23.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/>
        </svg>
      ),
      action: () =>
        window.open(
          `https://twitter.com/intent/tweet?text=${encodeURIComponent(shareText)}&url=${encodeURIComponent(currentUrl)}`,
          "_blank"
        ),
    },
    {
      name: "LinkedIn",
      icon: (
        <svg className="w-4 h-4 fill-current" viewBox="0 0 24 24">
          <path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z"/>
        </svg>
      ),
      action: () =>
        window.open(
          `https://www.linkedin.com/sharing/share-offsite/?url=${encodeURIComponent(currentUrl)}`,
          "_blank"
        ),
    },
    {
      name: "Facebook",
      icon: (
        <svg className="w-4 h-4 fill-current" viewBox="0 0 24 24">
          <path d="M24 12.073c0-6.627-5.373-12-12-12s-12 5.373-12 12c0 5.99 4.388 10.954 10.125 11.854v-8.385H7.078v-3.47h3.047V9.43c0-3.007 1.792-4.669 4.533-4.669 1.312 0 2.686.235 2.686.235v2.953H15.83c-1.491 0-1.956.925-1.956 1.874v2.25h3.328l-.532 3.47h-2.796v8.385C19.612 23.027 24 18.062 24 12.073z"/>
        </svg>
      ),
      action: () =>
        window.open(
          `https://www.facebook.com/sharer/sharer.php?u=${encodeURIComponent(currentUrl)}`,
          "_blank"
        ),
    },
  ];

  return (
    <div className="absolute right-0 top-full mt-2 w-64 bg-surface-container-lowest border border-surface-container-high/60 rounded-2xl shadow-xl p-3 z-50 animate-scale-in">
      <div className="flex items-center justify-between pb-2 mb-2 border-b border-surface-container-high/40">
        <span className="font-label-sm font-semibold text-on-surface text-xs uppercase tracking-wider">
          Share Fact Check
        </span>
        <button
          onClick={onClose}
          className="text-on-surface-variant hover:text-on-surface p-1 rounded-full cursor-pointer"
          type="button"
        >
          <MatIcon name="close" size={16} />
        </button>
      </div>

      <div className="space-y-1">
        {shareOptions.map((opt) => (
          <button
            key={opt.name}
            onClick={opt.action}
            className="w-full flex items-center gap-2.5 px-3 py-2 text-xs rounded-xl hover:bg-surface-container text-on-surface-variant hover:text-on-surface transition-colors cursor-pointer text-left"
            type="button"
          >
            {opt.icon}
            <span>{opt.name}</span>
          </button>
        ))}

        <button
          onClick={handleCopy}
          className="w-full flex items-center gap-2.5 px-3 py-2 text-xs rounded-xl hover:bg-surface-container text-primary font-medium transition-colors cursor-pointer text-left"
          type="button"
        >
          <MatIcon name={copied ? "check" : "content_copy"} size={16} />
          <span>{copied ? "Link Copied!" : "Copy Link"}</span>
        </button>
      </div>
    </div>
  );
}
