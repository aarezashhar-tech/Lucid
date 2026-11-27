import React from "react";
import { MatIcon } from "../common/MatIcon";

export function LoadingScreen({
  subject,
  progress,
  activeStep,
  onCancel,
}: {
  subject: string;
  progress: number;
  activeStep: number;
  onCancel: () => void;
}) {
  const steps = [
    { num: 1, label: "Reading content & OCR extraction" },
    { num: 2, label: "Generating AI summary & key takeaways" },
    { num: 3, label: "Extracting key factual claims" },
    { num: 4, label: "Corroborating independent wire evidence" },
    { num: 5, label: "Synthesizing consensus & verdict" },
  ];

  return (
    <div className="max-w-[680px] w-full mx-auto px-4 py-16 flex flex-col items-center text-center">
      {/* Official Lucid Logo with Subtle Pulse Animation */}
      <div className="mb-6 flex flex-col items-center">
        <img
          src="/logo-transparent.png"
          alt="Lucid Logo"
          className="h-16 md:h-20 w-auto object-contain animate-pulse"
        />
      </div>

      <div className="space-y-2 mb-8">
        <h2 className="font-headline-md text-headline-md text-on-surface font-semibold">
          Verifying Article Claims
        </h2>
        <p className="font-body-md text-body-md text-on-surface-variant max-w-md mx-auto truncate">
          “{subject}”
        </p>
      </div>

      {/* Progress Bar */}
      <div className="w-full bg-surface-container rounded-full h-2 mb-6 overflow-hidden">
        <div
          className="bg-primary h-full rounded-full transition-all duration-500 ease-out"
          style={{ width: `${progress}%` }}
        />
      </div>

      {/* Simple Stepper */}
      <div className="w-full lucid-card bg-surface-container-lowest rounded-xl p-5 border border-surface-container-high/40 shadow-xs text-left mb-6 space-y-3">
        {steps.map((st) => {
          const isDone = st.num < activeStep || progress === 100;
          const isCurrent = st.num === activeStep && progress < 100;
          return (
            <div key={st.num} className="flex items-center gap-3">
              <div
                className={`w-6 h-6 rounded-full flex items-center justify-center font-label-xs font-semibold ${
                  isDone
                    ? "bg-tertiary text-on-tertiary"
                    : isCurrent
                    ? "bg-primary text-on-primary animate-pulse"
                    : "bg-surface-container text-outline"
                }`}
              >
                {isDone ? <MatIcon name="check" size={14} /> : st.num}
              </div>
              <span
                className={`font-body-sm text-body-sm ${
                  isDone
                    ? "text-on-surface font-medium"
                    : isCurrent
                    ? "text-primary font-semibold"
                    : "text-outline"
                }`}
              >
                {st.label}
              </span>
            </div>
          );
        })}
      </div>

      <button
        onClick={onCancel}
        className="px-5 py-2 rounded-full bg-surface-container hover:bg-surface-container-high text-on-surface-variant font-label-sm text-label-sm transition-all cursor-pointer"
        type="button"
      >
        Cancel Verification
      </button>
    </div>
  );
}
