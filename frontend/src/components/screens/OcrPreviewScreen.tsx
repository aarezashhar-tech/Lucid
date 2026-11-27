import React, { useState } from "react";
import { MatIcon } from "../common/MatIcon";
import { LucidBrandLogo } from "../common/LucidBrandLogo";
import { OcrPreviewData } from "../../types/verification";

export function OcrPreviewScreen({
  previewData,
  onContinue,
  onCancel,
}: {
  previewData: OcrPreviewData;
  onContinue: (reviewedText: string) => void;
  onCancel: () => void;
}) {
  const [text, setText] = useState(previewData.extractedText || "");
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard?.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const previewSnippet = text.slice(0, 500);

  return (
    <div className="max-w-[760px] w-full mx-auto px-4 sm:px-6 py-12 md:py-16 flex flex-col items-center">
      {/* Brand Logo */}
      <div className="mb-4 flex flex-col items-center">
        <LucidBrandLogo size="md" showText={false} />
      </div>

      <div className="text-center max-w-xl mb-6 space-y-2">
        <h1 className="font-display-hero text-headline-lg font-bold text-on-surface">
          OCR Extracted Text
        </h1>
        <p className="font-body-md text-body-md text-on-surface-variant">
          Review extracted text from <strong className="text-on-surface font-semibold">{previewData.fileName}</strong> before verification.
        </p>
      </div>

      <div className="w-full lucid-card bg-surface-container-lowest rounded-xl shadow-md border border-surface-container-high/40 p-5 sm:p-6 flex flex-col gap-5 transition-all">
        {/* Header bar with confidence & quick copy */}
        <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-surface-container-high/30">
          <div className="flex items-center gap-2">
            <span className="w-7 h-7 rounded-full bg-primary/10 text-primary flex items-center justify-center font-label-xs font-bold">
              <MatIcon name="document_scanner" size={16} />
            </span>
            <span className="font-label-md text-label-md font-semibold text-on-surface">
              OCR Extracted Text (First 500 characters)
            </span>
          </div>

          <div className="flex items-center gap-2">
            <span
              className={`px-3 py-1 rounded-full font-label-xs text-label-xs font-semibold ${
                previewData.isLowConfidence
                  ? "bg-error/10 text-error border border-error/20"
                  : "bg-primary/10 text-primary"
              }`}
            >
              {previewData.isLowConfidence ? "Quality Warning" : `${Math.round(previewData.confidence)}% Confidence`}
            </span>

            <button
              onClick={handleCopy}
              className="flex items-center gap-1.5 px-3.5 py-2.5 min-h-[44px] rounded-full bg-surface-container hover:bg-surface-container-high text-on-surface font-label-xs text-xs font-semibold transition-colors cursor-pointer"
              type="button"
              aria-label="Copy extracted text to clipboard"
              title="Copy extracted text to clipboard"
            >
              <MatIcon name={copied ? "check" : "content_copy"} size={14} className={copied ? "text-tertiary" : ""} />
              <span>{copied ? "Copied" : "Copy text"}</span>
            </button>
          </div>
        </div>

        {/* Low Confidence Warning Alert */}
        {previewData.isLowConfidence && (
          <div className="p-4 rounded-lg bg-error-container text-on-error-container border border-error/30 flex items-start gap-3">
            <MatIcon name="warning" size={22} className="text-error shrink-0 mt-0.5" />
            <div className="text-body-sm space-y-1">
              <p className="font-semibold text-error">
                Text extraction quality may be low.
              </p>
              <p className="text-on-error-container/90">
                Some text may have been extracted inaccurately from the image. You can review and edit the text below before continuing.
              </p>
            </div>
          </div>
        )}

        {/* Text Review Box */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <label htmlFor="ocr-preview-textarea" className="font-label-xs text-xs text-on-surface-variant font-medium">
              Review text before verification:
            </label>
            <span className="font-label-xs text-xs text-outline">
              Showing first 500 characters ({previewSnippet.length}/500)
            </span>
          </div>

          <textarea
            id="ocr-preview-textarea"
            aria-label="Review and edit extracted OCR text"
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={8}
            className="w-full p-3.5 rounded-lg bg-surface-container-low border border-surface-container-high/40 font-mono text-body-sm text-on-surface outline-none focus:bg-surface-container-lowest focus:border-primary transition-all resize-y leading-relaxed"
            placeholder="No text was extracted from this image."
          />

          <div className="flex items-center justify-between text-outline text-label-xs px-1">
            <span>{text.length} total characters extracted</span>
            <span>Edit above to adjust OCR typos</span>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex flex-wrap items-center justify-between gap-3 pt-3 border-t border-surface-container-high/30">
          <button
            onClick={onCancel}
            type="button"
            className="px-4 py-2 rounded-full bg-surface-container hover:bg-surface-container-high text-on-surface-variant font-label-md text-label-md transition-colors cursor-pointer"
          >
            Cancel & Re-upload
          </button>

          <button
            onClick={() => onContinue(text)}
            type="button"
            disabled={!text.trim()}
            className="flex items-center gap-2 px-6 py-2.5 rounded-full bg-primary text-on-primary hover:bg-surface-tint disabled:opacity-40 font-label-md text-label-md font-semibold transition-all shadow-xs cursor-pointer"
          >
            <span>Continue Verification</span>
            <MatIcon name="arrow_forward" size={16} />
          </button>
        </div>
      </div>
    </div>
  );
}
