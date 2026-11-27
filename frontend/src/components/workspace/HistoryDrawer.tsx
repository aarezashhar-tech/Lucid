import React from "react";
import { MatIcon } from "../common/MatIcon";
import { PillBadge } from "../common/PillBadge";
import { HistoryItem } from "../../types/verification";

export function HistoryDrawer({
  isOpen,
  onClose,
  history,
  onSelectHistory,
  onClearHistory,
}: {
  isOpen: boolean;
  onClose: () => void;
  history: HistoryItem[];
  onSelectHistory: (item: HistoryItem) => void;
  onClearHistory?: () => void;
}) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 overflow-hidden animate-fade-in">
      {/* Backdrop */}
      <div
        onClick={onClose}
        className="absolute inset-0 bg-black/40 backdrop-blur-xs transition-opacity"
      />

      {/* Slide-over panel */}
      <div className="absolute inset-y-0 right-0 max-w-full flex pl-10">
        <div className="w-screen max-w-md bg-surface-container-lowest border-l border-surface-container-high shadow-2xl flex flex-col">
          {/* Header */}
          <div className="p-4 sm:p-6 border-b border-surface-container-high flex items-center justify-between">
            <div className="flex items-center gap-2">
              <MatIcon name="history" size={22} className="text-primary" />
              <h2 className="font-headline-sm text-base sm:text-lg font-bold text-on-surface">
                Verification History
              </h2>
            </div>
            <div className="flex items-center gap-2">
              {history.length > 0 && onClearHistory && (
                <button
                  onClick={onClearHistory}
                  className="px-2.5 py-1 text-xs text-error hover:bg-error-container/30 rounded-lg transition-colors cursor-pointer"
                  type="button"
                >
                  Clear All
                </button>
              )}
              <button
                onClick={onClose}
                className="p-1.5 text-on-surface-variant hover:text-on-surface hover:bg-surface-container rounded-full transition-colors cursor-pointer"
                type="button"
              >
                <MatIcon name="close" size={20} />
              </button>
            </div>
          </div>

          {/* List Content */}
          <div className="flex-1 overflow-y-auto p-4 sm:p-6 space-y-3">
            {history.length === 0 ? (
              <div className="h-full flex flex-col items-center justify-center text-center text-on-surface-variant gap-3 py-16">
                <div className="w-12 h-12 rounded-full bg-surface-container flex items-center justify-center text-outline">
                  <MatIcon name="manage_search" size={24} />
                </div>
                <div className="space-y-1">
                  <p className="font-headline-sm text-sm font-semibold text-on-surface">
                    No verifications yet
                  </p>
                  <p className="font-body-sm text-xs text-on-surface-variant max-w-xs">
                    Audited claims, articles, and OCR documents will appear here for instant replay.
                  </p>
                </div>
              </div>
            ) : (
              history.map((item) => (
                <div
                  key={item.id}
                  onClick={() => {
                    onSelectHistory(item);
                    onClose();
                  }}
                  className="p-3.5 rounded-xl border border-surface-container-high/60 bg-surface-container-low hover:bg-surface-container hover:border-primary/40 transition-all cursor-pointer space-y-2 group"
                >
                  <div className="flex items-center justify-between gap-2">
                    <PillBadge variant="verdict" status={String(item.verdict)} size="xs" />
                    <span className="font-mono text-[11px] text-outline">{item.time}</span>
                  </div>
                  <p className="font-headline-sm text-sm font-medium text-on-surface group-hover:text-primary transition-colors line-clamp-2 leading-snug">
                    {item.title}
                  </p>
                  <div className="flex items-center justify-between text-xs text-on-surface-variant pt-1 border-t border-surface-container-high/40">
                    <span>Trust Score: <strong className="text-on-surface font-semibold">{item.score}/100</strong></span>
                    <span className="text-primary font-medium flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
                      <span>View</span>
                      <MatIcon name="chevron_right" size={14} />
                    </span>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
