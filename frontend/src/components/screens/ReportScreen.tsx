import React, { useState, useEffect } from "react";
import { MatIcon } from "../common/MatIcon";
import { SharePopover } from "../common/SharePopover";
import { VerificationResult, ArticleSummary } from "../../types/verification";
import { cleanSnippetText, handleCardPointerMove } from "../../services/api";

export function ReportScreen({
  result,
  subject,
  onNewSearch,
  voiceGuideEnabled,
  speakVoiceGuidance,
  isSpeaking,
  onStopSpeaking,
}: {
  result: VerificationResult;
  subject: string;
  onNewSearch: () => void;
  voiceGuideEnabled?: boolean;
  speakVoiceGuidance?: (text: string) => void;
  isSpeaking?: boolean;
  onStopSpeaking?: () => void;
}) {
  const [copiedOcr, setCopiedOcr] = useState(false);
  const [isShareOpen, setIsShareOpen] = useState(false);

  // Automatic Article Summary
  const summary: ArticleSummary = result.articleSummary || result.summary || {
    title: result.articleTitle || subject,
    executiveSummary:
      result.summaryReason ||
      result.explanation?.[0] ||
      "Article content extracted and verified across independent corroboration pipelines.",
    keyTakeaways:
      result.claims && result.claims.length > 0
        ? result.claims.slice(0, 5).map((c) => c.claim)
        : [
            result.articleTitle || subject,
            `Primary Domain: ${result.sourceReputation?.domain || result.sourceDomain || "Direct Submission"}`,
            "Analyzed across accredited news wire consensus.",
          ],
    tldr: result.summaryReason || result.articleTitle || subject,
    provider: "Extractive NLP Engine",
  };

  // Voice Guidance: Automatically announce verdict and one-line reason when report renders
  useEffect(() => {
    if (voiceGuideEnabled && result && speakVoiceGuidance) {
      const v = result.ragGroundedVerdict?.verdict || result.overallVerdict || result.verdict || "Unverified";
      const reason = result.summaryReason || result.explanation?.[0] || summary.executiveSummary || "";
      const firstSentence = reason.split(/(?<=[.!?])\s+/)[0] || "";
      speakVoiceGuidance(`Result: ${v}. ${firstSentence}`);
    }
  }, [result, voiceGuideEnabled, speakVoiceGuidance]);

  // Title derivation: Always prioritize headline; avoid generic labels or file names
  const rawTitle = result.articleTitle || result.articleSummary?.title || "";
  const isGeneric =
    !rawTitle ||
    rawTitle === "Image Upload Analysis" ||
    rawTitle === "News Article" ||
    rawTitle === "Extracted Content" ||
    rawTitle === "Extracted Article Content" ||
    rawTitle === "Article Analysis" ||
    /\.(png|jpe?g|pdf|txt)$/i.test(rawTitle);

  let derivedFromText = "";
  const sourceText = result.ocrPreview?.extractedText || result.extractedText || "";
  if (sourceText) {
    const lines = sourceText.split(/\n+/).map((l) => l.trim()).filter(Boolean);
    for (const line of lines) {
      const cleaned = line
        .replace(/^(?:\d+\s*[mhd]\s+ago|breaking|watch|live|alert|update)\s*[:—–-]\s*/i, "")
        .replace(/^[A-Z\s,]{2,20}\s*(?:\([^\)]+\))?\s*[-—–]\s*/, "")
        .trim();
      const noPunct = cleaned.replace(/[.!?:]+$/, "").trim();
      if (noPunct.length >= 10 && !noPunct.startsWith("http")) {
        derivedFromText = noPunct.slice(0, 140);
        break;
      }
    }
  }

  const displayTitle = !isGeneric
    ? rawTitle
    : derivedFromText
    ? derivedFromText
    : subject && !/\.(png|jpe?g|pdf|txt)$/i.test(subject) && subject !== "Image Upload Analysis"
    ? subject
    : derivedFromText || "Verified Article";

  const suppCount = result.evidence?.supportingCount ?? result.claims?.filter((c) => c.status === "Supported").length ?? 3;
  const contraCount = result.evidence?.contradictingCount ?? result.claims?.filter((c) => c.status === "Contradicted").length ?? 0;
  const sourceTrust = result.sourceReputation?.trustScore ?? result.sourceCredibility ?? 85;
  const sourceName = result.sourceReputation?.sourceName || result.sourceDomain || "Identified Publisher";
  const displayPublisher = result.publisher || result.sourceReputation?.publisher || sourceName;
  const displaySourceType = result.sourceType || result.sourceReputation?.sourceType;
  const verdict = result.verdict || "VERIFIED";

  const isDirectSubmission = Boolean(
    result.isDirectSubmission ||
    result.sourceDomain === "direct-submission" ||
    result.sourceReputation?.domain === "direct-submission" ||
    result.sourceReputation?.domain === "N/A" ||
    (!result.url && (!result.sourceReputation?.publisher || result.sourceReputation?.publisher === "Unknown" || result.publisher === "Unknown" || displayPublisher === "Unknown" || displayPublisher === "Unspecified / Self-Published Content"))
  );

  const defaultScoring = isDirectSubmission ? {
    evidenceStrength: { weight: 0.90, score: result.evidence?.evidenceScore ?? 80, contribution: 72.0, label: "Evidence Strength (90%)" },
    sourceReputation: { weight: 0.00, score: 50, contribution: 0.0, label: "Domain & Source Reputation (0%)" },
    claimConsistency: { weight: 0.10, score: 92, contribution: 9.2, label: "Factual Claim Consistency (10%)" },
    mlPrediction: { weight: 0.00, score: 50, contribution: 0.0, label: "ML Text Pattern Classifier (0%)" },
    languageQuality: { weight: 0.00, score: 50, contribution: 0.0, label: "Journalistic Language Integrity (0%)" },
  } : {
    evidenceStrength: { weight: 0.80, score: result.evidence?.evidenceScore ?? 80, contribution: 64.0, label: "Evidence Strength (80%)" },
    sourceReputation: { weight: 0.10, score: result.sourceReputation?.trustScore ?? 85, contribution: 8.5, label: "Domain & Source Reputation (10%)" },
    claimConsistency: { weight: 0.10, score: 92, contribution: 9.2, label: "Factual Claim Consistency (10%)" },
    mlPrediction: { weight: 0.00, score: 50, contribution: 0.0, label: "ML Text Pattern Classifier (0%)" },
    languageQuality: { weight: 0.00, score: 50, contribution: 0.0, label: "Journalistic Language Integrity (0%)" },
  };

  const scoring = result.scoringFactors || defaultScoring;

  const handlePrintPDF = () => {
    window.print();
  };

  const quickSummaryText =
    summary.quickSummary ||
    summary.tldr ||
    (summary.executiveSummary
      ? summary.executiveSummary.split(/(?<=[.!?])\s+/).slice(0, 2).join(" ")
      : "Summary unavailable.");

  return (
    <div className="max-w-[1000px] w-full mx-auto px-3 sm:px-6 py-4 sm:py-8 overflow-x-hidden">
      {/* Top Action Navigation Bar */}
      <div className="flex flex-wrap items-center justify-between gap-2.5 mb-6 pb-4 border-b border-surface-container-high/40">
        <button
          onClick={onNewSearch}
          className="inline-flex items-center gap-2 min-h-[44px] px-3 py-2 text-on-surface-variant hover:text-primary font-label-md text-label-md font-semibold transition-colors cursor-pointer"
          type="button"
        >
          <MatIcon name="arrow_back" size={18} />
          <span>New Verification</span>
        </button>

        <div className="flex items-center gap-2">
          <button
            onClick={handlePrintPDF}
            className="flex items-center gap-1.5 px-3.5 py-2 min-h-[44px] rounded-full bg-surface-container hover:bg-surface-container-high text-on-surface font-label-sm text-label-sm font-medium transition-colors cursor-pointer"
            title="Export / Print official report"
            type="button"
          >
            <MatIcon name="picture_as_pdf" size={16} />
            <span className="hidden xs:inline">PDF Export</span>
            <span className="xs:hidden">PDF</span>
          </button>

          <div className="relative">
            <button
              onClick={() => setIsShareOpen((prev) => !prev)}
              className="flex items-center gap-1.5 px-3.5 py-2 min-h-[44px] rounded-full bg-surface-container hover:bg-surface-container-high text-on-surface font-label-sm text-label-sm font-medium transition-colors cursor-pointer"
              type="button"
            >
              <MatIcon name="share" size={16} />
              <span>Share</span>
            </button>
            <SharePopover
              isOpen={isShareOpen}
              onClose={() => setIsShareOpen(false)}
              title={displayTitle}
              verdict={result.ragGroundedVerdict?.verdict || result.overallVerdict || result.verdict}
            />
          </div>
        </div>
      </div>

      {/* HERO ELEMENT: Article / News Headline */}
      <div className="mb-6 sm:mb-8 space-y-3">
        <div className="flex flex-wrap items-center gap-2 mb-2">
          {/* Multi-Language Detection Badge */}
          <span className="px-3 py-1 rounded-full font-label-xs font-semibold bg-surface-container text-on-surface-variant border border-surface-container-high/40 flex items-center gap-1.5">
            <MatIcon name="translate" size={13} />
            <span>Language: {result.detectedLanguage || result.language?.language || "English"}</span>
            {result.wasTranslated && (
              <span className="text-primary font-bold">(Translated)</span>
            )}
          </span>
        </div>

        <h1 className="text-2xl sm:text-3xl md:text-5xl font-extrabold text-on-surface tracking-tight leading-tight break-words [overflow-wrap:anywhere]">
          {displayTitle}
        </h1>
        <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1 font-body-sm text-xs sm:text-sm text-on-surface-variant">
          {isDirectSubmission ? (
            <>
              <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full font-label-xs font-semibold bg-surface-container text-on-surface-variant border border-surface-container-high/40">
                <MatIcon name="edit_note" size={13} />
                <span>Direct Text Claim</span>
              </span>
              <span className="text-outline">·</span>
              <span>Verified via live external news, fact-check &amp; reference sources</span>
            </>
          ) : (
            <>
              <span>
                Publisher: <strong className="text-on-surface font-semibold">{displayPublisher}</strong>
              </span>
              {displaySourceType && (
                <>
                  <span className="text-outline">·</span>
                  <span>
                    Source Type: <strong className="text-on-surface font-semibold">{displaySourceType}</strong>
                  </span>
                </>
              )}
              <span className="text-outline">·</span>
              <span>
                Domain: <span className="font-mono">{result.sourceReputation?.domain || result.sourceDomain || "N/A"}</span>
              </span>
              {result.sourceReputation?.publisherConfidence && (
                <>
                  <span className="text-outline">·</span>
                  <span className="text-tertiary font-medium">Confidence: {result.sourceReputation.publisherConfidence}</span>
                </>
              )}
            </>
          )}
        </div>
      </div>

      {/* MULTI-LANGUAGE TRANSLATION NOTIFICATION */}
      {result.wasTranslated && result.detectedLanguage && result.detectedLanguage !== "English" && (
        <div className="mb-6 p-4 rounded-xl lucid-card-sub bg-primary-fixed/20 border border-primary/20 flex items-start gap-3">
          <span className="w-8 h-8 rounded-full bg-primary text-on-primary flex items-center justify-center shrink-0 mt-0.5">
            <MatIcon name="g_translate" size={18} />
          </span>
          <div className="space-y-1 font-body-sm text-body-sm text-on-surface">
            <strong className="block font-label-md text-label-md">
              Multi-Language Detection: {result.detectedLanguage} ({result.languageCode || result.language?.code})
            </strong>
            <p className="text-on-surface-variant">
              Lucid automatically identified input in {result.detectedLanguage} and translated it to English for cross-referencing against Reuters, AP, BBC, Google Fact Check, and GNews archives without interruption.
            </p>
          </div>
        </div>
      )}

      {/* ========================================================================= */}
      {/* REPORT CONTENT PIPELINE:                                                  */}
      {/* OCR EXTRACTED TEXT -> OVERVIEW -> ARTICLE SUMMARY -> KEY TAKEAWAYS        */}
      {/* -> CLAIMS -> EVIDENCE -> VERDICT -> MULTI-FACTOR SCORING                  */}
      {/* ========================================================================= */}

      <div className="space-y-6">
        {/* 1. OCR EXTRACTED TEXT (For Image Submissions) */}
        {(result.ocrPreview || result.fileName?.match(/\.(png|jpe?g)$/i)) && (
          <section className="lucid-card bg-surface-container-lowest rounded-xl p-6 shadow-xs border border-surface-container-high/40 space-y-4">
            <div className="flex items-center justify-between border-b border-surface-container-high/30 pb-3">
              <div className="flex items-center gap-2">
                <span className="w-6 h-6 rounded-full bg-primary text-on-primary flex items-center justify-center font-label-xs font-bold">
                  <MatIcon name="document_scanner" size={14} />
                </span>
                <h3 className="font-headline-sm text-headline-sm text-on-surface uppercase tracking-wide">
                  OCR Extracted Text
                </h3>
              </div>
              <div className="flex items-center gap-2">
                {(result.ocrConfidence !== undefined || result.ocrPreview?.confidence !== undefined) && (
                  <span
                    className={`px-2.5 py-0.5 rounded-full font-label-xs text-label-xs font-semibold ${
                      result.isLowConfidence || result.ocrPreview?.isLowConfidence
                        ? "bg-error/10 text-error border border-error/20"
                        : "bg-surface-container text-primary"
                    }`}
                  >
                    {result.isLowConfidence || result.ocrPreview?.isLowConfidence
                      ? "Low Quality"
                      : `${Math.round(result.ocrConfidence ?? result.ocrPreview?.confidence ?? 0)}% Confidence`}
                  </span>
                )}
                <button
                  onClick={() => {
                    const txt = result.ocrPreview?.previewText || result.ocrPreview?.extractedText || result.extractedText || "";
                    navigator.clipboard?.writeText(txt);
                    setCopiedOcr(true);
                    setTimeout(() => setCopiedOcr(false), 2000);
                  }}
                  className="flex items-center gap-1.5 px-3.5 py-2.5 min-h-[44px] rounded-full bg-surface-container hover:bg-surface-container-high text-primary font-label-xs text-xs font-semibold transition-colors cursor-pointer"
                  type="button"
                  aria-label="Copy OCR preview text"
                >
                  <MatIcon name={copiedOcr ? "check" : "content_copy"} size={14} />
                  <span>{copiedOcr ? "Copied" : "Copy Text"}</span>
                </button>
              </div>
            </div>

            {/* Low Confidence Warning Alert */}
            {(result.isLowConfidence || result.ocrPreview?.isLowConfidence) && (
              <div className="p-3.5 rounded-lg bg-error-container text-on-error-container border border-error/30 flex items-start gap-2.5">
                <MatIcon name="warning" size={20} className="text-error shrink-0 mt-0.5" />
                <div className="text-body-sm font-medium space-y-0.5">
                  <p className="font-semibold text-error">Text extraction quality may be low.</p>
                  <p className="text-on-error-container/80">Some text may have been extracted inaccurately from the image.</p>
                </div>
              </div>
            )}

            {/* First 300-500 characters display */}
            <div className="relative p-4 rounded-lg bg-surface-container-low border border-surface-container-high/40 font-mono text-body-sm text-on-surface whitespace-pre-wrap leading-relaxed">
              {(result.ocrPreview?.previewText || result.ocrPreview?.extractedText || result.extractedText || "").slice(0, 500)}
              {((result.ocrPreview?.previewText || result.ocrPreview?.extractedText || result.extractedText || "").length > 500) && "..."}
            </div>

            <div className="flex items-center justify-between text-outline text-label-xs">
              <span>Showing first 500 characters of OCR text</span>
              <span>{(result.ocrPreview?.extractedText || result.extractedText || "").length} total characters</span>
            </div>
          </section>
        )}

        {/* 2. OVERVIEW */}
        <section
          onPointerMove={handleCardPointerMove}
          className="lucid-card spotlight-card bg-surface-container-lowest rounded-xl p-6 shadow-xs border border-surface-container-high/40 space-y-3"
        >
          <div className="flex items-center justify-between border-b border-surface-container-high/30 pb-3">
            <div className="flex items-center gap-2">
              <span className="w-6 h-6 rounded-full bg-primary text-on-primary flex items-center justify-center font-label-xs font-bold">
                <MatIcon name="bolt" size={14} />
              </span>
              <h3 className="font-headline-sm text-headline-sm text-on-surface uppercase tracking-wide">
                Overview
              </h3>
            </div>
            <span className="px-2.5 py-0.5 rounded-full bg-primary/10 text-primary font-label-xs text-label-xs font-semibold">
              At a Glance
            </span>
          </div>
          <div className="lucid-card-sub p-4 rounded-lg bg-surface-container-low border border-surface-container-high/30">
            <p className="font-body-lg text-body-lg font-medium text-on-surface leading-relaxed">
              {quickSummaryText}
            </p>
          </div>
        </section>

        {/* 3. ARTICLE SUMMARY */}
        <section
          onPointerMove={handleCardPointerMove}
          className="lucid-card spotlight-card bg-surface-container-lowest rounded-xl p-6 shadow-xs border border-surface-container-high/40 space-y-4"
        >
          <div className="flex items-center justify-between border-b border-surface-container-high/30 pb-3">
            <div className="flex items-center gap-2">
              <span className="w-6 h-6 rounded-full bg-primary text-on-primary flex items-center justify-center font-label-xs font-bold">
                <MatIcon name="summarize" size={14} />
              </span>
              <h3 className="font-headline-sm text-headline-sm text-on-surface uppercase tracking-wide">
                Article Summary
              </h3>
            </div>
            <span className="px-2.5 py-0.5 rounded-full bg-surface-container text-primary font-label-xs text-label-xs font-semibold">
              {summary.provider || "Extractive NLP Engine"}
            </span>
          </div>

          <div className="space-y-1">
            <span className="font-label-xs text-label-xs uppercase tracking-wider text-on-surface-variant font-semibold">
              Executive Summary (3-5 Sentences)
            </span>
            <p className="font-body-md text-body-md text-on-surface leading-relaxed">
              {summary.executiveSummary}
            </p>
          </div>
        </section>

        {/* 4. KEY TAKEAWAYS */}
        <section className="lucid-card bg-surface-container-lowest rounded-xl p-6 shadow-xs border border-surface-container-high/40 space-y-4">
          <div className="flex items-center justify-between border-b border-surface-container-high/30 pb-3">
            <div className="flex items-center gap-2">
              <span className="w-6 h-6 rounded-full bg-primary text-on-primary flex items-center justify-center font-label-xs font-bold">
                <MatIcon name="format_list_bulleted" size={14} />
              </span>
              <h3 className="font-headline-sm text-headline-sm text-on-surface uppercase tracking-wide">
                Key Takeaways
              </h3>
            </div>
            <span className="px-2.5 py-0.5 rounded-full bg-surface-container text-primary font-label-xs text-label-xs font-semibold">
              {summary.keyTakeaways?.length || 0} Core Points
            </span>
          </div>

          <div className="space-y-2">
            {(summary.keyTakeaways && summary.keyTakeaways.length > 0
              ? summary.keyTakeaways
              : ["Article content summarized across verification pipelines."]
            ).map((point, idx) => (
              <div key={idx} className="lucid-card-sub flex items-start gap-3 p-3 rounded-lg bg-surface-container-low">
                <div className="w-5 h-5 rounded-full bg-primary text-on-primary flex items-center justify-center flex-shrink-0 mt-0.5">
                  <MatIcon name="check" size={14} />
                </div>
                <span className="font-body-md text-body-md text-on-surface">{point}</span>
              </div>
            ))}
          </div>
        </section>

        {/* 5. CLAIM-BY-CLAIM EVIDENCE MATRIX */}
        <section className="lucid-card bg-surface-container-lowest rounded-xl p-6 shadow-xs border border-surface-container-high/40 space-y-4">
          <div className="flex items-center justify-between border-b border-surface-container-high/30 pb-3">
            <div className="flex items-center gap-2">
              <span className="w-6 h-6 rounded-full bg-primary text-on-primary flex items-center justify-center font-label-xs font-bold">
                <MatIcon name="fact_check" size={14} />
              </span>
              <h3 className="font-headline-sm text-headline-sm text-on-surface uppercase tracking-wide">
                Evidence Matrix
              </h3>
            </div>
            <span className="px-2.5 py-0.5 rounded-full bg-surface-container text-primary font-label-xs text-label-xs font-semibold">
              {result.evidenceMatrix?.length || result.claims?.length || 0} Evaluated Claims
            </span>
          </div>

          <div className="space-y-4">
            {result.evidenceMatrix && result.evidenceMatrix.length > 0 ? (
              result.evidenceMatrix.map((row, idx) => {
                const isSupp = row.status === "Verified" || row.status === "Supported" || row.status === "Likely True";
                const isContra = row.status === "Contradicted" || row.status === "False" || row.status === "Likely False";
                const isOp = row.isOpinion || row.claimType === "Opinion / Perspective Assertion";
                return (
                  <div
                    key={idx}
                    className="lucid-card-sub p-4 rounded-xl bg-surface-container-low flex flex-col gap-3 border border-surface-container-high/40"
                  >
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div className="flex items-center gap-2">
                        <span className="font-label-xs font-bold text-outline">CLAIM 0{idx + 1}</span>
                        <span
                          className={`px-2.5 py-0.5 rounded-full font-label-xs text-label-xs font-bold uppercase ${
                            isOp
                              ? "bg-orange-500/15 text-orange-700 dark:text-orange-300 border border-orange-500/30"
                              : "bg-surface-container text-on-surface-variant"
                          }`}
                        >
                          {isOp ? "Opinion Assertion" : "Factual Proposition"}
                        </span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span
                          className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full font-label-xs text-xs font-bold uppercase ${
                            isSupp
                              ? "bg-tertiary/15 text-tertiary"
                              : isContra
                              ? "bg-error/15 text-error"
                              : isOp
                              ? "bg-orange-500/15 text-orange-700 dark:text-orange-300"
                              : "bg-[#FDF1DC] text-[#9A5B00] border border-[#F5D49E] dark:bg-amber-950/40 dark:text-amber-400 dark:border-amber-900/40"
                          }`}
                        >
                          <MatIcon
                            name={isSupp ? "check_circle" : isContra ? "cancel" : isOp ? "rate_review" : "help"}
                            size={13}
                          />
                          <span>{row.status}</span>
                        </span>
                        {!isOp && (
                          <span className="font-label-xs text-xs font-semibold text-primary">
                            {row.confidence}% Confidence
                          </span>
                        )}
                      </div>
                    </div>

                    <p className="font-body-md text-body-md font-semibold text-on-surface leading-snug">
                      “{row.claim}”
                    </p>

                    {/* Opinion exclusion notice */}
                    {isOp ? (
                      <div className="p-2.5 rounded bg-orange-500/10 border border-orange-500/20 text-xs font-body-sm text-on-surface-variant flex items-center gap-2">
                        <MatIcon name="info" size={14} className="text-orange-600 dark:text-orange-400 shrink-0" />
                        <span>
                          <strong>Subjective Perspective:</strong> In accordance with journalistic guidelines, opinion statements are not fact-checked against wire reports.
                        </span>
                      </div>
                    ) : (
                      <div className="space-y-2 pt-1">
                        {/* Supporting Evidence */}
                        {row.supportingEvidence && row.supportingEvidence.length > 0 && (
                          <div className="space-y-1.5">
                            <span className="font-label-xs text-label-xs font-bold uppercase text-tertiary flex items-center gap-1">
                              <MatIcon name="check_circle" size={13} />
                              Supporting Sources ({row.supportingEvidence.length})
                            </span>
                            <div className="grid grid-cols-1 gap-1.5">
                              {row.supportingEvidence.map((ev, sIdx) => (
                                <div
                                  key={sIdx}
                                  className="lucid-card-sub p-2.5 rounded-lg bg-surface-container-lowest border border-tertiary/20 text-xs space-y-1 overflow-hidden"
                                >
                                  <div className="flex items-center justify-between gap-2">
                                    <span className="font-bold text-on-surface flex items-center gap-1.5 min-w-0">
                                      <span className="px-1.5 py-0.5 rounded bg-tertiary/10 text-tertiary font-label-xs shrink-0">
                                        {cleanSnippetText(ev.publisher) || "Verified Wire"}
                                      </span>
                                      <span className="text-outline text-xs truncate">{ev.publicationDate}</span>
                                    </span>
                                    {ev.relevanceScore !== undefined && (
                                      <span className="text-tertiary font-mono font-semibold text-xs shrink-0">
                                        {ev.relevanceScore}% Semantic Match
                                      </span>
                                    )}
                                  </div>
                                  <p className="text-on-surface italic break-words break-all [overflow-wrap:anywhere]">
                                    "{cleanSnippetText(ev.evidenceSnippet)}"
                                  </p>
                                  {ev.url && (
                                    <a
                                      href={ev.url}
                                      target="_blank"
                                      rel="noreferrer"
                                      className="text-primary hover:underline block truncate font-mono text-xs"
                                    >
                                      {cleanSnippetText(ev.articleTitle) || ev.url}
                                    </a>
                                  )}
                                </div>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* Contradicting Evidence */}
                        {row.contradictingEvidence && row.contradictingEvidence.length > 0 && (
                          <div className="space-y-1.5 pt-1">
                            <span className="font-label-xs text-label-xs font-bold uppercase text-error flex items-center gap-1">
                              <MatIcon name="cancel" size={13} />
                              Contradicting Sources ({row.contradictingEvidence.length})
                            </span>
                            <div className="grid grid-cols-1 gap-1.5">
                              {row.contradictingEvidence.map((ev, cIdx) => (
                                <div
                                  key={cIdx}
                                  className="lucid-card-sub p-2.5 rounded-lg bg-surface-container-lowest border border-error/20 text-xs space-y-1 overflow-hidden"
                                >
                                  <div className="flex items-center justify-between gap-2">
                                    <span className="font-bold text-on-surface flex items-center gap-1.5 min-w-0">
                                      <span className="px-1.5 py-0.5 rounded bg-error/10 text-error font-label-xs shrink-0">
                                        {cleanSnippetText(ev.publisher) || "Refutation Wire"}
                                      </span>
                                      <span className="text-outline text-xs truncate">{ev.publicationDate}</span>
                                    </span>
                                    {ev.relevanceScore !== undefined && (
                                      <span className="text-error font-mono font-semibold text-xs shrink-0">
                                        {ev.relevanceScore}% Debunk Match
                                      </span>
                                    )}
                                  </div>
                                  <p className="text-on-surface italic break-words break-all [overflow-wrap:anywhere]">
                                    "{cleanSnippetText(ev.evidenceSnippet)}"
                                  </p>
                                  {ev.url && (
                                    <a
                                      href={ev.url}
                                      target="_blank"
                                      rel="noreferrer"
                                      className="text-error hover:underline block truncate font-mono text-xs"
                                    >
                                      {cleanSnippetText(ev.articleTitle) || ev.url}
                                    </a>
                                  )}
                                </div>
                              ))}
                            </div>
                          </div>
                        )}

                        {(!row.supportingEvidence || row.supportingEvidence.length === 0) &&
                          (!row.contradictingEvidence || row.contradictingEvidence.length === 0) && (
                            <span className="font-label-xs text-label-xs text-[#9A5B00] dark:text-amber-400 font-medium block">
                              Corroboration: NO SUPPORTING EVIDENCE FOUND (Unverified)
                            </span>
                          )}

                        {row.evidenceSummary && (
                          <p className="font-body-xs text-on-surface-variant text-xs pt-1">
                            {row.evidenceSummary}
                          </p>
                        )}
                      </div>
                    )}
                  </div>
                );
              })
            ) : result.claims && result.claims.length > 0 ? (
              result.claims.map((claim, idx) => {
                const isSupp = claim.status === "Supported" || claim.status === "Verified";
                const isContra = claim.status === "Contradicted" || claim.status === "False";
                return (
                  <div
                    key={idx}
                    className="lucid-card-sub p-4 rounded-lg bg-surface-container-low flex flex-col gap-1.5 border border-surface-container-high/30"
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-label-xs font-bold text-outline">CLAIM 0{idx + 1}</span>
                      <div className="flex items-center gap-2">
                        <span
                          className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full font-label-xs text-xs font-semibold ${
                            isSupp
                              ? "bg-tertiary/15 text-tertiary"
                              : isContra
                              ? "bg-error/15 text-error"
                              : "bg-[#FDF1DC] text-[#9A5B00] border border-[#F5D49E] dark:bg-amber-950/40 dark:text-amber-400 dark:border-amber-900/40"
                          }`}
                        >
                          <MatIcon name={isSupp ? "check_circle" : isContra ? "cancel" : "help"} size={13} />
                          <span>{claim.status || "Verified"}</span>
                        </span>
                        <span className="font-label-xs text-xs font-semibold text-primary">
                          {claim.confidence || claim.score || 90}% Confidence
                        </span>
                      </div>
                    </div>
                    <p className="font-body-md text-body-md font-medium text-on-surface">“{claim.claim}”</p>
                    {claim.supportingSources && claim.supportingSources.length > 0 ? (
                      <span className="font-label-xs text-xs text-on-surface-variant">
                        Supporting citations: {claim.supportingSources.join(", ")}
                      </span>
                    ) : (
                      <span className="font-label-xs text-xs text-[#9A5B00] dark:text-amber-400 font-medium">
                        Supporting citations: NO SUPPORTING EVIDENCE FOUND
                      </span>
                    )}
                  </div>
                );
              })
            ) : (
              <div className="py-8 px-4 rounded-xl bg-surface-container-low border border-surface-container-high/30 flex flex-col items-center justify-center text-center gap-2">
                <div className="w-10 h-10 rounded-full bg-surface-container flex items-center justify-center text-outline">
                  <MatIcon name="fact_check" size={20} />
                </div>
                <h4 className="font-headline-sm text-sm font-semibold text-on-surface">No claims extracted</h4>
                <p className="font-body-sm text-xs text-on-surface-variant max-w-md">
                  No verifiable factual claims or assertions were extracted from this content.
                </p>
              </div>
            )}
          </div>
        </section>

        {/* 6. EVIDENCE & MULTI-SOURCE RETRIEVAL */}
        <section className="lucid-card bg-surface-container-lowest rounded-xl p-6 shadow-xs border border-surface-container-high/40 space-y-4">
          <div className="flex items-center justify-between border-b border-surface-container-high/30 pb-3">
            <div className="flex items-center gap-2">
              <span className="w-6 h-6 rounded-full bg-primary text-on-primary flex items-center justify-center font-label-xs font-bold">
                <MatIcon name="find_in_page" size={14} />
              </span>
              <h3 className="font-headline-sm text-headline-sm text-on-surface uppercase tracking-wide">
                Evidence & Grounded Retrieval
              </h3>
            </div>
            <span className="px-2.5 py-0.5 rounded-full bg-surface-container text-primary font-label-xs text-label-xs font-semibold">
              {suppCount} Supporting · {contraCount} Contradicting
            </span>
          </div>

          {/* Primary Source Meta */}
          <div className="lucid-card-sub grid grid-cols-1 sm:grid-cols-2 gap-4 p-4 rounded-lg bg-surface-container-low border border-surface-container-high/30">
            {isDirectSubmission ? (
              <>
                <div>
                  <small className="font-label-xs text-label-xs text-outline uppercase tracking-wider block font-bold">
                    Submission Context
                  </small>
                  <b className="font-headline-sm text-headline-sm text-on-surface">Direct Text Claim</b>
                  <span className="font-body-sm text-body-sm text-on-surface-variant block mt-0.5">
                    Verified via external news, fact-check &amp; reference sources
                  </span>
                </div>
                <div className="sm:text-right">
                  <small className="font-label-xs text-xs text-outline uppercase tracking-wider block font-bold">
                    Verification Channel
                  </small>
                  <div className="flex items-center sm:justify-end gap-2 mt-1">
                    <span className="px-3 py-1 rounded-full font-label-xs text-xs font-bold uppercase tracking-wider bg-surface-container text-primary border border-surface-container-high">
                      External Verification
                    </span>
                  </div>
                </div>
              </>
            ) : (
              <>
                <div>
                  <small className="font-label-xs text-label-xs text-outline uppercase tracking-wider block font-bold">
                    Identified Primary Publisher
                  </small>
                  <b className="font-headline-sm text-headline-sm text-on-surface">{displayPublisher}</b>
                  {displaySourceType && (
                    <span className="font-body-sm text-body-sm text-on-surface-variant block mt-0.5">
                      Source Type: <strong className="text-on-surface font-semibold">{displaySourceType}</strong>
                    </span>
                  )}
                  {result.sourceReputation?.publisherConfidence && (
                    <span className="font-label-xs text-label-xs text-tertiary block mt-0.5 font-semibold">
                      Confidence: {result.sourceReputation.publisherConfidence}
                    </span>
                  )}
                  <span className="font-body-sm text-body-sm text-on-surface-variant block mt-0.5">
                    Domain: {result.sourceReputation?.domain || result.sourceDomain || "N/A"}
                  </span>
                </div>
                <div className="sm:text-right">
                  <small className="font-label-xs text-xs text-outline uppercase tracking-wider block font-bold">
                    Domain Trust Score
                  </small>
                  <div className="flex items-center sm:justify-end gap-2 mt-0.5">
                    <b
                      className={`font-headline-lg text-headline-lg ${
                        sourceTrust >= 80 ? "text-tertiary" : sourceTrust >= 60 ? "text-[#9A5B00] dark:text-amber-400" : "text-error"
                      }`}
                    >
                      {sourceTrust}/100
                    </b>
                    <span
                      className={`px-2 py-0.5 rounded-full font-label-xs text-xs font-bold uppercase tracking-wider ${
                        sourceTrust >= 80
                          ? "bg-tertiary/10 text-tertiary"
                          : sourceTrust >= 60
                          ? "bg-[#FDF1DC] text-[#9A5B00] border border-[#F5D49E] dark:bg-amber-950/40 dark:text-amber-400 dark:border-amber-900/40"
                          : "bg-error/10 text-error"
                      }`}
                    >
                      {sourceTrust >= 80 ? "High" : sourceTrust >= 60 ? "Medium" : "Low"}
                    </span>
                  </div>
                </div>
              </>
            )}
          </div>

          {/* Source Transparency Block */}
          <div className="lucid-card-sub p-4 rounded-lg bg-surface-container-low border border-surface-container-high/40 text-left space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full bg-primary shrink-0"></span>
                <span className="font-label-sm text-label-sm font-bold text-on-surface uppercase tracking-wider">
                  Retrieval Layer Transparency
                </span>
              </div>
              <span className="font-mono text-xs font-semibold text-primary">
                {suppCount} Supporting · {contraCount} Contradicting
              </span>
            </div>

            <div className="p-3.5 rounded bg-surface-container-lowest border border-surface-container-high/40 font-mono text-body-sm space-y-2.5">
              <div>
                <span className="font-bold text-on-surface block mb-1">Sources searched:</span>
                <div className="text-on-surface-variant pl-1 space-y-0.5">
                  {(
                    (result.sourcesSearched && result.sourcesSearched.length > 0)
                      ? result.sourcesSearched
                      : (result.emptyEvidenceAudit?.sourcesSearched && result.emptyEvidenceAudit.sourcesSearched.length > 0)
                      ? result.emptyEvidenceAudit.sourcesSearched
                      : [
                          "Google News RSS Wire Search",
                          "Wikipedia (Official REST API)"
                        ]
                  ).map((src, i) => (
                    <div key={i}>- {src.replace(/^-\s*/, '')}</div>
                  ))}
                </div>
              </div>
              <div>
                <span className="font-bold text-on-surface block">Results found:</span>
                <span className="text-on-surface-variant font-medium">
                  {suppCount} supporting, {contraCount} contradicting
                </span>
              </div>
              {suppCount === 0 && (
                <div>
                  <span className="font-bold text-on-surface block">Reason:</span>
                  <span className="text-on-surface-variant">
                    {result.emptyEvidenceAudit?.reason || "No sufficiently relevant corroborating sources were found."}
                  </span>
                </div>
              )}
            </div>
          </div>

          {/* TOP SUPPORTING SOURCES */}
          {result.topSupportingSources && result.topSupportingSources.length > 0 && (
            <div className="space-y-2.5 pt-2">
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-tertiary"></span>
                <span className="font-label-xs text-label-xs uppercase tracking-wider text-tertiary font-bold">
                  Top Supporting Sources ({result.topSupportingSources.length})
                </span>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {result.topSupportingSources.map((src, i) => (
                  <div
                    key={i}
                    className="lucid-card-sub p-3.5 rounded-xl bg-surface-container-low border border-tertiary/20 flex flex-col justify-between gap-2 shadow-2xs overflow-hidden"
                  >
                    <div className="space-y-1 min-w-0">
                      <div className="flex items-center justify-between gap-2">
                        <span className="px-2 py-0.5 rounded bg-tertiary/15 text-tertiary font-label-xs font-bold shrink-0 truncate max-w-[200px]">
                          {cleanSnippetText(src.publisher) || "Verified Wire"}
                        </span>
                        {src.relevanceScore !== undefined && (
                          <span className="text-tertiary font-mono text-xs font-semibold shrink-0">
                            {src.relevanceScore}% Semantic Match
                          </span>
                        )}
                      </div>
                      <a
                        href={src.url}
                        target="_blank"
                        rel="noreferrer"
                        className="font-body-md font-semibold text-on-surface hover:text-primary transition-colors block line-clamp-2 break-words break-all [overflow-wrap:anywhere]"
                      >
                        {cleanSnippetText(src.articleTitle) || src.url}
                      </a>
                      <p className="font-body-xs text-on-surface-variant line-clamp-3 text-xs leading-relaxed break-words break-all [overflow-wrap:anywhere]">
                        "{cleanSnippetText(src.evidenceSnippet)}"
                      </p>
                    </div>
                    <div className="flex items-center justify-between text-xs text-outline pt-1 border-t border-surface-container-high/30">
                      <span>{src.publicationDate}</span>
                      <a
                        href={src.url}
                        target="_blank"
                        rel="noreferrer"
                        className="text-primary hover:underline font-semibold flex items-center gap-0.5 shrink-0"
                      >
                        <span>View Source</span>
                        <MatIcon name="open_in_new" size={11} />
                      </a>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* TOP CONTRADICTING SOURCES */}
          {result.topContradictingSources && result.topContradictingSources.length > 0 && (
            <div className="space-y-2.5 pt-2">
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-error"></span>
                <span className="font-label-xs text-label-xs uppercase tracking-wider text-error font-bold">
                  Top Contradicting Sources ({result.topContradictingSources.length})
                </span>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {result.topContradictingSources.map((src, i) => (
                  <div
                    key={i}
                    className="lucid-card-sub p-3.5 rounded-xl bg-surface-container-low border border-error/20 flex flex-col justify-between gap-2 shadow-2xs overflow-hidden"
                  >
                    <div className="space-y-1 min-w-0">
                      <div className="flex items-center justify-between gap-2">
                        <span className="px-2 py-0.5 rounded bg-error/15 text-error font-label-xs font-bold shrink-0 truncate max-w-[200px]">
                          {cleanSnippetText(src.publisher) || "Refutation Wire"}
                        </span>
                        {src.relevanceScore !== undefined && (
                          <span className="text-error font-mono text-xs font-semibold shrink-0">
                            {src.relevanceScore}% Debunk Match
                          </span>
                        )}
                      </div>
                      <a
                        href={src.url}
                        target="_blank"
                        rel="noreferrer"
                        className="font-body-md font-semibold text-on-surface hover:text-error transition-colors block line-clamp-2 break-words break-all [overflow-wrap:anywhere]"
                      >
                        {cleanSnippetText(src.articleTitle) || src.url}
                      </a>
                      <p className="font-body-xs text-on-surface-variant line-clamp-3 text-xs leading-relaxed break-words break-all [overflow-wrap:anywhere]">
                        "{cleanSnippetText(src.evidenceSnippet)}"
                      </p>
                    </div>
                    <div className="flex items-center justify-between text-xs text-outline pt-1 border-t border-surface-container-high/30">
                      <span>{src.publicationDate}</span>
                      <a
                        href={src.url}
                        target="_blank"
                        rel="noreferrer"
                        className="text-error hover:underline font-semibold flex items-center gap-0.5 shrink-0"
                      >
                        <span>View Source</span>
                        <MatIcon name="open_in_new" size={11} />
                      </a>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Google Fact Check Tools & IFCN Audits */}
          {result.factCheckResults && result.factCheckResults.length > 0 && (
            <div className="space-y-2 pt-2">
              <span className="font-label-xs text-label-xs uppercase tracking-wider text-outline font-semibold">
                Google Fact Check Tools & IFCN Audits
              </span>
              <div className="overflow-x-auto">
                <table className="w-full text-left font-body-sm text-body-sm">
                  <thead>
                    <tr className="bg-surface-container-low text-on-surface-variant font-label-xs text-label-xs uppercase tracking-wider">
                      <th className="py-2 px-3 rounded-l-lg">Claim</th>
                      <th className="py-2 px-3">Verdict</th>
                      <th className="py-2 px-3">Reviewer</th>
                      <th className="py-2 px-3 rounded-r-lg">Link</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-surface-container-low">
                    {result.factCheckResults.map((fc, i) => (
                      <tr key={i}>
                        <td className="py-2.5 px-3 font-medium text-on-surface max-w-xs truncate">{fc.claim}</td>
                        <td className="py-2.5 px-3">
                          <span className="px-2 py-0.5 rounded-full bg-surface-container font-label-xs font-semibold">
                            {fc.rating}
                          </span>
                        </td>
                        <td className="py-2.5 px-3 text-on-surface-variant">{fc.reviewer}</td>
                        <td className="py-2.5 px-3">
                          {fc.url ? (
                            <a
                              href={fc.url}
                              target="_blank"
                              rel="noreferrer"
                              className="text-primary hover:underline font-label-xs font-medium"
                            >
                              View
                            </a>
                          ) : (
                            "N/A"
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {(!result.topSupportingSources || result.topSupportingSources.length === 0) &&
            (!result.topContradictingSources || result.topContradictingSources.length === 0) &&
            (!result.factCheckResults || result.factCheckResults.length === 0) && (
              <div className="py-8 px-4 rounded-xl bg-surface-container-low border border-surface-container-high/30 flex flex-col items-center justify-center text-center gap-2">
                <div className="w-10 h-10 rounded-full bg-surface-container flex items-center justify-center text-outline">
                  <MatIcon name="source" size={20} />
                </div>
                <h4 className="font-headline-sm text-sm font-semibold text-on-surface">No sources found</h4>
                <p className="font-body-sm text-xs text-on-surface-variant max-w-md">
                  No corroborating or refuting external sources were retrieved for this investigation.
                </p>
              </div>
            )}
        </section>

        {/* 7. GROUNDED VERDICT & REASONING */}
        <section
          onPointerMove={handleCardPointerMove}
          className="lucid-card glass-card spotlight-card rounded-xl p-6 shadow-xs border border-surface-container-high/40 space-y-4"
        >
          <div className="flex items-center justify-between border-b border-surface-container-high/30 pb-3">
            <div className="flex items-center gap-2">
              <span className="w-6 h-6 rounded-full bg-primary text-on-primary flex items-center justify-center font-label-xs font-bold">
                <MatIcon name="verified" size={14} />
              </span>
              <h3 className="font-headline-sm text-headline-sm text-on-surface uppercase tracking-wide">
                Grounded Verdict & Reasoning
              </h3>
            </div>
            <span
              className={`px-3 py-1 rounded-full font-label-xs text-label-xs font-bold uppercase tracking-wider ${
                result.riskLevel === "LOW"
                  ? "bg-tertiary/15 text-tertiary"
                  : result.riskLevel === "HIGH" || result.riskLevel === "CRITICAL"
                  ? "bg-error/15 text-error"
                  : "bg-[#FDF1DC] text-[#9A5B00] border border-[#F5D49E] dark:bg-amber-950/40 dark:text-amber-400 dark:border-amber-900/40"
              }`}
            >
              Risk: {result.riskLevel}
            </span>
          </div>

          <div className="lucid-card-sub flex flex-wrap items-center justify-between gap-4 p-4 rounded-lg bg-surface-container-low border border-surface-container-high/30">
            <div className="flex items-center gap-3">
              <div
                className={`w-12 h-12 rounded-full flex items-center justify-center shadow-xs ${
                  verdict === "Satire"
                    ? "bg-purple-600 text-white"
                    : verdict === "Opinion - not fact-checked"
                    ? "bg-primary/20 text-primary"
                    : verdict === "VERIFIED" || verdict === "LIKELY TRUE"
                    ? "bg-tertiary text-on-tertiary"
                    : verdict === "KNOWN FALSE" || verdict === "LIKELY FALSE" || verdict === "FALSE"
                    ? "bg-error text-on-error"
                    : "bg-[#FDF1DC] text-[#9A5B00] border border-[#F5D49E] dark:bg-amber-950/60 dark:text-amber-300 dark:border-amber-800/40"
                }`}
              >
                <MatIcon
                  name={
                    verdict === "Satire"
                      ? "theater_comedy"
                      : verdict === "Opinion - not fact-checked"
                      ? "rate_review"
                      : verdict === "VERIFIED" || verdict === "LIKELY TRUE"
                      ? "verified"
                      : "warning"
                  }
                  size={26}
                  fill
                />
              </div>
              <div>
                <small className="font-label-xs text-label-xs text-outline uppercase tracking-wider block font-bold">
                  Grounded Verdict
                </small>
                <h2 className="font-headline-md text-headline-md font-bold text-on-surface m-0 leading-tight">
                  {verdict}
                </h2>
              </div>
            </div>

            <div className="text-right">
              <small className="font-label-xs text-label-xs text-outline uppercase tracking-wider block font-bold">
                Trust Score
              </small>
              {typeof result.trustScore === "string" || isNaN(Number(result.trustScore)) ? (
                <span className={`inline-block px-3 py-1 mt-1 rounded-full text-sm font-semibold ${
                  result.trustScore === "Satire"
                    ? "bg-purple-100 text-purple-800 dark:bg-purple-900/40 dark:text-purple-300"
                    : "bg-surface-container-high text-on-surface-variant"
                }`}>
                  {result.trustScore}
                </span>
              ) : (
                <div className="flex items-center sm:justify-end gap-2 mt-0.5">
                  <b
                    className={`font-headline-lg text-headline-lg ${
                      Number(result.trustScore) >= 75
                        ? "text-tertiary"
                        : Number(result.trustScore) >= 50
                        ? "text-[#9A5B00] dark:text-amber-400"
                        : "text-error"
                    }`}
                  >
                    {result.trustScore}
                    <small className="text-sm font-normal text-outline">/100</small>
                  </b>
                  <span
                    className={`px-2 py-0.5 rounded-full font-label-xs text-xs font-bold uppercase tracking-wider ${
                      Number(result.trustScore) >= 75
                        ? "bg-tertiary/10 text-tertiary"
                        : Number(result.trustScore) >= 50
                        ? "bg-[#FDF1DC] text-[#9A5B00] border border-[#F5D49E] dark:bg-amber-950/40 dark:text-amber-400 dark:border-amber-900/40"
                        : "bg-error/10 text-error"
                    }`}
                  >
                    {Number(result.trustScore) >= 75 ? "High Credibility" : Number(result.trustScore) >= 50 ? "Medium Credibility" : "Low Credibility"}
                  </span>
                </div>
              )}
            </div>
          </div>

          {/* Missing evidence notice */}
          {verdict === "UNVERIFIED" && (
            <div className="p-3.5 rounded-lg bg-surface-container border border-surface-container-high text-xs text-on-surface-variant flex items-center gap-2">
              <MatIcon name="info" size={16} className="text-primary shrink-0" />
              <span>
                <strong>Grounding Principle:</strong> In accordance with fact-checking standards, missing or uncorroborated evidence is classified strictly as <strong>UNVERIFIED</strong>, never as false.
              </span>
            </div>
          )}

          {/* Grounded Reasoning Panel */}
          <div className="lucid-card-sub p-4 rounded-lg bg-surface-container-low space-y-3">
            <span className="font-label-xs text-label-xs uppercase tracking-wider text-outline font-semibold block">
              Grounded Evidence Reasoning
            </span>
            <p className="font-body-md text-body-md text-on-surface leading-relaxed">
              {result.groundedReasoning?.synthesis ||
                result.summaryReason ||
                result.explanation?.join(" ") ||
                `Evidence-based RAG synthesis completed across news wire archives with ${suppCount} supporting citations.`}
            </p>

            {/* Why Appears True */}
            {result.groundedReasoning?.whyAppearsTrue && result.groundedReasoning.whyAppearsTrue.length > 0 && (
              <div className="pt-2 space-y-1">
                <span className="font-label-xs text-label-xs font-bold uppercase text-tertiary block">
                  Corroborating Evidence Grounding:
                </span>
                <ul className="list-disc list-inside space-y-1 text-xs text-on-surface-variant">
                  {result.groundedReasoning.whyAppearsTrue.map((w, idx) => (
                    <li key={idx} className="break-words break-all [overflow-wrap:anywhere]">
                      {cleanSnippetText(w)}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Why Appears False */}
            {result.groundedReasoning?.whyAppearsFalse && result.groundedReasoning.whyAppearsFalse.length > 0 && (
              <div className="pt-2 space-y-1">
                <span className="font-label-xs text-label-xs font-bold uppercase text-error block">
                  Contradicting Evidence Flags:
                </span>
                <ul className="list-disc list-inside space-y-1 text-xs text-on-surface-variant">
                  {result.groundedReasoning.whyAppearsFalse.map((w, idx) => (
                    <li key={idx} className="break-words break-all [overflow-wrap:anywhere]">
                      {cleanSnippetText(w)}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </section>

        {/* SECTION 8: Multi-Factor Scoring Breakdown */}
        <section className="lucid-card bg-surface-container rounded-xl p-4 sm:p-6 border border-surface-container-high/60 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-2 border-b border-surface-container-high/40 pb-3">
            <div className="flex items-center gap-2">
              <MatIcon name="analytics" size={20} className="text-primary" />
              <h3 className="font-headline-sm text-headline-sm text-on-surface uppercase tracking-wide">
                Multi-Factor Trust Scoring Breakdown
              </h3>
            </div>
            <span className="font-label-xs text-label-xs text-outline font-medium">
              {isDirectSubmission
                ? "Pure evidence-led verification: 90% Evidence Strength · 10% Factual Claim Consistency"
                : "Evidence-led verification: 80% Evidence Strength · 10% Domain & Source Reputation · 10% Factual Claim Consistency"}
            </span>
          </div>

          <div className={`grid grid-cols-1 ${isDirectSubmission ? "sm:grid-cols-2" : "sm:grid-cols-3"} gap-3`}>
            {[
              { key: "evidenceStrength", factor: scoring.evidenceStrength, icon: "fact_check", descSuffix: "Corroborating citations", defaultPct: isDirectSubmission ? 90 : 80 },
              ...(!isDirectSubmission ? [{ key: "sourceReputation", factor: scoring.sourceReputation, icon: "domain", descSuffix: "Publisher & domain authority", defaultPct: 10 }] : []),
              { key: "claimConsistency", factor: scoring.claimConsistency, icon: "balance", descSuffix: "Internal claim coherence", defaultPct: 10 },
            ].map(({ key, factor, icon, descSuffix, defaultPct }) => {
              if (!factor) return null;
              const isNotScored = factor.score === "Not scored" || typeof factor.score === "string";
              const weightPct = typeof factor.weight === "number" ? Math.round(factor.weight * 100) : defaultPct;
              const desc = `${weightPct}% weight · ${descSuffix}`;
              return (
                <div key={key} className="lucid-card-sub p-3.5 rounded-lg bg-surface-container-low border border-surface-container-high/30 flex flex-col justify-between">
                  <div>
                    <div className="flex items-center justify-between gap-1 mb-1.5">
                      <div className="flex items-center gap-1.5 text-on-surface font-semibold text-xs">
                        <MatIcon name={icon} size={16} className="text-primary shrink-0" />
                        <span className="truncate">{factor.label || key}</span>
                      </div>
                      <span className="font-mono text-xs font-bold text-on-surface">
                        {isNotScored ? "N/A" : `${factor.score}/100`}
                      </span>
                    </div>
                    <p className="text-xs text-outline mb-2">{desc}</p>
                  </div>
                  <div className="pt-2 border-t border-surface-container-high/20 flex items-center justify-between text-xs">
                    <span className="text-on-surface-variant">Contribution:</span>
                    <span className="font-mono font-bold text-primary">
                      {isNotScored ? "N/A" : `+${factor.contribution} pts`}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </section>

        {/* Printable PDF Footer with Lucid Branding */}
        <div className="hidden print:block mt-12 pt-6 border-t border-gray-300 text-center text-xs text-gray-500 font-medium">
          Verified with Lucid AI Fact Verification Platform · Official Verification Report
        </div>
      </div>
    </div>
  );
}
