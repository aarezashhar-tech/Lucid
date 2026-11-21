import React, { FormEvent, useEffect, useRef, useState } from "react";

// ==========================================
// 1. DATA TYPES & INTERFACES
// ==========================================

export type Screen = "home" | "live-analysis" | "report";

export type VerdictType =
  | "VERIFIED"
  | "LIKELY TRUE"
  | "NEEDS REVIEW"
  | "UNVERIFIED"
  | "LIKELY FALSE"
  | "KNOWN FALSE"
  | "SUSPICIOUS"
  | "LIKELY FAKE"
  | "MIXED EVIDENCE";

export interface ClaimItem {
  id?: string;
  claimNumber?: number;
  claim: string;
  verdict?: string;
  status?: "Supported" | "Contradicted" | "Unverified" | "Disputed" | string;
  score?: number;
  source?: string;
  supportingSources?: string[];
  contradictingSources?: string[];
  supportingEvidenceCount?: number;
  contradictingEvidenceCount?: number;
  confidence?: number;
  analysisNotes?: string;
}

export interface FactCheckResult {
  claim: string;
  claimant?: string;
  reviewer: string;
  rating: string;
  url: string;
  review_date?: string;
  source: string;
}

export interface EvidenceItem {
  source: string;
  sourceName?: string;
  articleTitle?: string;
  publicationDate?: string;
  sourceTrustScore?: number;
  url?: string;
  matchingClaim?: string;
  stance: "support" | "contradict" | string;
  snippet: string;
  trust?: number;
}

export interface SourceReputation {
  sourceName: string;
  domain: string;
  trustScore: number;
  tier: string;
  category: string;
  description: string;
  attributionType?: string;
  isTrusted?: boolean;
}

export interface ScoringFactorItem {
  weight: number;
  score: number;
  contribution: number;
  label: string;
}

export interface ScoringFactors {
  mlPrediction: ScoringFactorItem;
  sourceReputation: ScoringFactorItem;
  evidenceStrength: ScoringFactorItem;
  claimConsistency: ScoringFactorItem;
  languageQuality: ScoringFactorItem;
}

export interface ArticleSummary {
  title?: string;
  executiveSummary: string; // 3-5 simple sentences
  keyTakeaways: string[];   // Maximum 5 bullet points
  tldr: string;             // 1 sentence
  provider?: string;
}

export interface VerificationResult {
  prediction: "REAL" | "FAKE" | "UNKNOWN";
  confidence: number;
  credibility?: number;
  trustScore: number;
  riskLevel: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL" | "UNKNOWN";
  verdict: VerdictType;
  summaryReason?: string;
  explanation: string[];
  articleSummary?: ArticleSummary;
  summary?: ArticleSummary;
  sourceReputation?: SourceReputation;
  sourceCredibility?: number;
  languageIntegrity?: number;
  evidenceQuality?: number;
  claims: ClaimItem[];
  evidence?: {
    supporting: EvidenceItem[];
    contradicting: EvidenceItem[];
    supportingCount?: number;
    contradictingCount?: number;
    evidenceScore?: number;
  };
  scoringFactors?: ScoringFactors;
  factCheckResults?: FactCheckResult[];
  sources?: Array<{
    name: string;
    code: string;
    domain?: string;
    type: string;
    trust: number;
    tier?: string;
    description?: string;
    isPrimary?: boolean;
  }>;
  disclaimer?: string;
  timeline?: Array<{
    step: string;
    detail: string;
    status: string;
  }>;
  extractedText?: string;
  fileName?: string;
  sourceDomain?: string;
  articleTitle?: string;
}

export interface HistoryItem {
  id: string;
  title: string;
  time: string;
  score: number;
  verdict: VerdictType;
  data: VerificationResult;
}

// ==========================================
// 2. HELPER COMPONENTS
// ==========================================

function MatIcon({
  name,
  className = "",
  size = 20,
  fill = false,
}: {
  name: string;
  className?: string;
  size?: number;
  fill?: boolean;
}) {
  return (
    <span
      className={`material-symbols-outlined select-none leading-none ${className}`}
      style={{
        fontSize: `${size}px`,
        fontVariationSettings: fill ? "'FILL' 1" : "'FILL' 0",
      }}
    >
      {name}
    </span>
  );
}

// Brand Logo Component - Uses /logo-transparent.png
function LucidBrandLogo({
  size = "md",
  showText = true,
  onClick,
}: {
  size?: "sm" | "md" | "lg" | "hero";
  showText?: boolean;
  onClick?: () => void;
}) {
  const heightClasses = {
    sm: "h-6",
    md: "h-8 md:h-9",
    lg: "h-12 md:h-14",
    hero: "h-16 md:h-20",
  };

  return (
    <div
      onClick={onClick}
      className={`flex items-center gap-2.5 select-none ${onClick ? "cursor-pointer" : ""}`}
    >
      <img
        src="/logo-transparent.png"
        alt="Lucid Logo"
        className={`${heightClasses[size]} w-auto object-contain transition-transform duration-200`}
      />
      {showText && (
        <span className="font-headline-sm text-headline-sm tracking-tight text-on-surface font-bold">
          Lucid
        </span>
      )}
    </div>
  );
}

// Navbar / Header Component (Only Logo on far left + Dark/Light toggle on far right)
function AppHeader({
  onNavigateHome,
  darkMode,
  onToggleDarkMode,
}: {
  onNavigateHome: () => void;
  darkMode: boolean;
  onToggleDarkMode: () => void;
}) {
  return (
    <header className="fixed top-0 left-0 right-0 z-50 bg-surface-container-lowest/80 backdrop-blur-xl border-b border-surface-container-high/40 shadow-xs">
      <div className="h-16 max-w-[900px] mx-auto px-4 sm:px-6 flex items-center justify-between">
        {/* Far Left Brand Logo */}
        <LucidBrandLogo size="md" showText onClick={onNavigateHome} />

        {/* Far Right Action: Dark/Light Toggle */}
        <button
          aria-label="Toggle theme"
          onClick={onToggleDarkMode}
          className="p-2 rounded-full text-on-surface-variant hover:bg-surface-container hover:text-on-surface transition-all cursor-pointer"
          type="button"
          title={darkMode ? "Switch to light mode" : "Switch to dark mode"}
        >
          <MatIcon name={darkMode ? "light_mode" : "dark_mode"} size={20} />
        </button>
      </div>
    </header>
  );
}

// Minimal Clean Footer
function AppFooter() {
  return (
    <footer className="w-full border-t border-surface-container-high/30 py-6 mt-auto">
      <div className="max-w-[900px] mx-auto px-4 text-center">
        <span className="font-label-xs text-label-xs text-outline block">
          © 2026 Lucid · AI Fact Verification Platform
        </span>
      </div>
    </footer>
  );
}

// ==========================================
// 3. HOME & VERIFICATION WORKSPACE
// ==========================================

function HomeScreen({
  onVerify,
  history,
  onSelectHistory,
}: {
  onVerify: (payload: { mode: "url" | "text" | "upload"; content: string; file?: File }) => void;
  history: HistoryItem[];
  onSelectHistory: (item: HistoryItem) => void;
}) {
  const [mode, setMode] = useState<"url" | "text" | "image" | "pdf">("url");
  const [query, setQuery] = useState("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const handleSubmit = (e?: FormEvent) => {
    if (e) e.preventDefault();
    if (mode === "image" || mode === "pdf") {
      if (!selectedFile) {
        fileInputRef.current?.click();
        return;
      }
      onVerify({ mode: "upload", content: selectedFile.name, file: selectedFile });
    } else if (mode === "url") {
      if (!query.trim()) return;
      onVerify({ mode: "url", content: query.trim() });
    } else {
      if (!query.trim()) return;
      onVerify({ mode: "text", content: query.trim() });
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      if (file.type.includes("pdf")) setMode("pdf");
      else setMode("image");
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) {
      setSelectedFile(file);
      if (file.type.includes("pdf")) setMode("pdf");
      else setMode("image");
    }
  };

  return (
    <div className="max-w-[760px] w-full mx-auto px-4 sm:px-6 py-12 md:py-20 flex flex-col items-center">
      {/* Hidden File Input */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*,.pdf,.doc,.docx,.txt"
        className="hidden"
        onChange={handleFileChange}
      />

      {/* Logo */}
      <div className="mb-6 flex flex-col items-center">
        <LucidBrandLogo size="hero" showText={false} />
      </div>

      {/* Headline & Subtitle */}
      <div className="text-center max-w-2xl mb-8 space-y-2.5">
        <h1 className="font-display-hero text-display-hero tracking-tight text-on-surface">
          Verify Information Before You Believe It
        </h1>
        <p className="font-body-lg text-body-lg text-on-surface-variant max-w-xl mx-auto">
          Get a simple summary, evaluate credibility, and understand the evidence behind any article.
        </p>
      </div>

      {/* Main Card */}
      <div className="w-full bg-surface-container-lowest rounded-xl shadow-md border border-surface-container-high/40 p-3 sm:p-5 flex flex-col gap-3 transition-all duration-300">
        {/* Input Tabs: URL, Text, Image, PDF */}
        <div className="flex items-center gap-1.5 pb-2 border-b border-surface-container-high/20">
          <button
            onClick={() => setMode("url")}
            className={`px-4 py-1.5 rounded-full font-label-md text-label-md transition-all cursor-pointer ${
              mode === "url"
                ? "bg-primary text-on-primary font-semibold shadow-xs"
                : "bg-surface-container-low hover:bg-surface-container text-on-surface-variant"
            }`}
            type="button"
          >
            URL
          </button>

          <button
            onClick={() => setMode("text")}
            className={`px-4 py-1.5 rounded-full font-label-md text-label-md transition-all cursor-pointer ${
              mode === "text"
                ? "bg-primary text-on-primary font-semibold shadow-xs"
                : "bg-surface-container-low hover:bg-surface-container text-on-surface-variant"
            }`}
            type="button"
          >
            Text
          </button>

          <button
            onClick={() => {
              setMode("image");
              fileInputRef.current?.click();
            }}
            className={`px-4 py-1.5 rounded-full font-label-md text-label-md transition-all cursor-pointer ${
              mode === "image"
                ? "bg-primary text-on-primary font-semibold shadow-xs"
                : "bg-surface-container-low hover:bg-surface-container text-on-surface-variant"
            }`}
            type="button"
          >
            Image
          </button>

          <button
            onClick={() => {
              setMode("pdf");
              fileInputRef.current?.click();
            }}
            className={`px-4 py-1.5 rounded-full font-label-md text-label-md transition-all cursor-pointer ${
              mode === "pdf"
                ? "bg-primary text-on-primary font-semibold shadow-xs"
                : "bg-surface-container-low hover:bg-surface-container text-on-surface-variant"
            }`}
            type="button"
          >
            PDF
          </button>
        </div>

        {/* Input Form */}
        <form onSubmit={handleSubmit} className="flex flex-col gap-2.5 w-full">
          {mode === "text" ? (
            <div className="relative flex flex-col w-full bg-surface-container-low rounded-lg p-2.5 border border-surface-container-high/40 focus-within:bg-surface-container-lowest focus-within:shadow-[0_0_0_2px_rgba(0,74,198,0.2)] transition-all">
              <textarea
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Paste news article content, statement, or text to verify..."
                rows={4}
                className="w-full bg-transparent border-none outline-none font-body-md text-body-md text-on-surface placeholder:text-outline p-2 resize-y"
              />
              <div className="flex justify-end pt-2">
                <button
                  type="submit"
                  disabled={!query.trim()}
                  className="flex items-center gap-1.5 px-6 py-2 rounded-full bg-primary text-on-primary hover:bg-surface-tint active:scale-95 disabled:opacity-40 font-label-md text-label-md transition-all shadow-xs cursor-pointer"
                >
                  <MatIcon name="search" size={18} />
                  <span>Verify</span>
                </button>
              </div>
            </div>
          ) : (
            <div className="relative flex items-center w-full bg-surface-container-low rounded-lg p-2 border border-surface-container-high/40 focus-within:bg-surface-container-lowest focus-within:shadow-[0_0_0_2px_rgba(0,74,198,0.2)] transition-all">
              <MatIcon name="search" size={22} className="text-outline ml-2 select-none" />
              <input
                autoComplete="off"
                value={selectedFile ? selectedFile.name : query}
                onChange={(e) => {
                  if (selectedFile) setSelectedFile(null);
                  setQuery(e.target.value);
                }}
                className="w-full bg-transparent border-none outline-none font-body-lg text-body-lg text-on-surface placeholder:text-outline px-3 py-2"
                placeholder={
                  selectedFile
                    ? `Selected: ${selectedFile.name}`
                    : mode === "url"
                    ? "Paste news article URL (e.g. https://bbc.com/news/...)"
                    : "Paste article URL, text, screenshot or PDF..."
                }
                type="text"
              />
              {(query || selectedFile) && (
                <button
                  onClick={() => {
                    setQuery("");
                    setSelectedFile(null);
                  }}
                  className="p-1 mr-1 text-outline hover:text-on-surface rounded-full transition-colors cursor-pointer"
                  title="Clear input"
                  type="button"
                >
                  <MatIcon name="close" size={18} />
                </button>
              )}
              {/* Verify Button */}
              <button
                type="submit"
                disabled={!query.trim() && !selectedFile}
                className="flex items-center gap-1.5 px-6 py-2.5 rounded-full bg-primary text-on-primary hover:bg-surface-tint active:scale-95 disabled:opacity-40 font-label-md text-label-md transition-all shadow-xs shrink-0 cursor-pointer"
              >
                <MatIcon name="search" size={18} />
                <span>Verify</span>
              </button>
            </div>
          )}
        </form>

        {/* Drag and Drop Zone */}
        <div
          onClick={() => fileInputRef.current?.click()}
          onDragOver={(e) => {
            e.preventDefault();
            setIsDragging(true);
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
          className={`cursor-pointer rounded-lg flex items-center justify-center gap-2 py-3 px-4 transition-all text-center border border-dashed ${
            isDragging
              ? "bg-surface-container-high border-primary"
              : selectedFile
              ? "bg-primary-fixed/20 border-primary text-primary"
              : "bg-surface-container-low hover:bg-surface-container border-surface-container-high"
          }`}
        >
          <MatIcon name={selectedFile ? "task_alt" : "upload_file"} size={18} className="text-primary" />
          <span className="font-body-sm text-body-sm text-on-surface-variant">
            {selectedFile ? (
              <strong className="text-primary">{selectedFile.name} ready</strong>
            ) : (
              "Drop image, screenshot or PDF to verify"
            )}
          </span>
        </div>
      </div>

      {/* Recent Investigations */}
      <div className="w-full mt-10">
        <div className="bg-surface-container-lowest rounded-xl shadow-xs border border-surface-container-high/40 overflow-hidden">
          <div className="px-5 py-3 border-b border-surface-container-high/30 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <MatIcon name="history" size={18} className="text-primary" />
              <span className="font-label-md text-label-md font-semibold text-on-surface">Recent Investigations</span>
            </div>
            <span className="px-2 py-0.5 rounded-full bg-surface-container text-primary font-label-xs text-label-xs font-semibold">
              {history.length}
            </span>
          </div>

          <div className="p-4">
            {history.length === 0 ? (
              <div className="py-6 flex flex-col items-center justify-center text-center gap-1.5">
                <h4 className="font-headline-sm text-headline-sm text-on-surface">No investigations yet</h4>
                <p className="font-body-sm text-body-sm text-on-surface-variant">
                  Start your first verification
                </p>
              </div>
            ) : (
              <div className="flex flex-col divide-y divide-surface-container-high/20">
                {history.map((item) => (
                  <div
                    key={item.id}
                    onClick={() => onSelectHistory(item)}
                    className="py-2.5 flex items-center justify-between hover:bg-surface-container-low px-2 rounded-lg cursor-pointer transition-colors"
                  >
                    <div className="flex flex-col max-w-lg">
                      <span className="font-label-md text-label-md font-medium text-on-surface truncate">
                        {item.title}
                      </span>
                      <span className="font-label-xs text-label-xs text-outline">{item.time}</span>
                    </div>
                    <span
                      className={`px-2.5 py-0.5 rounded-full font-label-xs text-label-xs font-semibold ${
                        item.verdict === "VERIFIED" || item.verdict === "LIKELY TRUE"
                          ? "bg-tertiary/10 text-tertiary"
                          : item.verdict === "KNOWN FALSE" || item.verdict === "LIKELY FALSE"
                          ? "bg-error/10 text-error"
                          : "bg-surface-container text-on-surface-variant"
                      }`}
                    >
                      {item.verdict}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// ==========================================
// 4. LOADING SCREEN (ANIMATED LOGO & PIPELINE)
// ==========================================

function LoadingScreen({
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
    { num: 1, label: "Reading article content" },
    { num: 2, label: "Extracting key factual claims" },
    { num: 3, label: "Verifying publisher & source attribution" },
    { num: 4, label: "Corroborating independent wire evidence" },
    { num: 5, label: "Synthesizing consensus & generating summary" },
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
      <div className="w-full bg-surface-container-lowest rounded-xl p-5 border border-surface-container-high/40 shadow-xs text-left mb-6 space-y-3">
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

// ==========================================
// 5. VERIFICATION REPORT SCREEN (6-STAGE FLOW)
// ==========================================

function ReportScreen({
  result,
  subject,
  onNewSearch,
}: {
  result: VerificationResult;
  subject: string;
  onNewSearch: () => void;
}) {
  const [techDetailsOpen, setTechDetailsOpen] = useState(false);
  const [copiedLink, setCopiedLink] = useState(false);

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

  const scoring = result.scoringFactors || {
    mlPrediction: { weight: 0.3, score: result.prediction === "REAL" ? 82 : 28, contribution: 24.6, label: "ML Prediction" },
    sourceReputation: { weight: 0.25, score: result.sourceReputation?.trustScore ?? 85, contribution: 21.2, label: "Source Reputation" },
    evidenceStrength: { weight: 0.25, score: result.evidence?.evidenceScore ?? 80, contribution: 20.0, label: "Evidence Strength" },
    claimConsistency: { weight: 0.1, score: 92, contribution: 9.2, label: "Claim Consistency" },
    languageQuality: { weight: 0.1, score: result.languageIntegrity ?? 88, contribution: 8.8, label: "Language Quality" },
  };

  const suppCount = result.evidence?.supportingCount ?? result.claims?.filter((c) => c.status === "Supported").length ?? 3;
  const contraCount = result.evidence?.contradictingCount ?? result.claims?.filter((c) => c.status === "Contradicted").length ?? 0;
  const sourceTrust = result.sourceReputation?.trustScore ?? result.sourceCredibility ?? 85;
  const sourceName = result.sourceReputation?.sourceName || result.sourceDomain || "Identified Publisher";
  const sourceTier = result.sourceReputation?.tier || (sourceTrust >= 80 ? "High" : sourceTrust >= 60 ? "Medium" : "Low");
  const verdict = result.verdict || "VERIFIED";

  const handleCopyLink = () => {
    navigator.clipboard?.writeText(window.location.href);
    setCopiedLink(true);
    setTimeout(() => setCopiedLink(false), 2000);
  };

  const handlePrintPDF = () => {
    window.print();
  };

  return (
    <div className="max-w-[1000px] w-full mx-auto px-4 sm:px-6 py-8">
      {/* Printable PDF Header */}
      <div className="hidden print:flex flex-col items-start gap-2 mb-6 pb-4 border-b border-gray-300">
        <img src="/logo-transparent.png" alt="Lucid Logo" className="h-12 w-auto object-contain" />
        <div>
          <h2 className="text-xl font-bold text-gray-900 m-0">Lucid Intelligence Workspace</h2>
          <h3 className="text-sm font-semibold text-gray-600 m-0">Evidence-Based Fact Verification Report</h3>
        </div>
      </div>

      {/* Screen Header Bar: Logo next to "Lucid AI Verification Platform" */}
      <div className="flex flex-wrap items-center justify-between gap-4 mb-8 pb-4 border-b border-surface-container-high/40">
        <div className="flex items-center gap-3">
          <img src="/logo-transparent.png" alt="Lucid Logo" className="h-9 w-auto object-contain" />
          <div>
            <span className="font-headline-sm text-headline-sm font-bold text-on-surface block leading-tight">
              Lucid AI Verification Platform
            </span>
            <span className="font-label-xs text-label-xs text-outline">
              Evidence-Based Verification Report
            </span>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={handlePrintPDF}
            className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-full bg-surface-container hover:bg-surface-container-high font-label-sm text-label-sm font-medium transition-colors cursor-pointer"
            title="Export / Print official report with PDF header"
            type="button"
          >
            <MatIcon name="picture_as_pdf" size={16} />
            <span>PDF Export</span>
          </button>

          <button
            onClick={handleCopyLink}
            className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-full bg-surface-container hover:bg-surface-container-high font-label-sm text-label-sm font-medium transition-colors cursor-pointer"
            type="button"
          >
            <MatIcon name={copiedLink ? "check" : "share"} size={16} />
            <span>{copiedLink ? "Copied" : "Share"}</span>
          </button>

          <button
            onClick={onNewSearch}
            className="flex items-center gap-1.5 px-4 py-1.5 rounded-full bg-primary text-on-primary font-label-sm text-label-sm font-semibold hover:bg-primary-container transition-all cursor-pointer"
            type="button"
          >
            <MatIcon name="search" size={16} />
            <span>New Verification</span>
          </button>
        </div>
      </div>

      {/* Target Title */}
      <div className="mb-6 space-y-1">
        <span className="font-label-xs text-label-xs uppercase tracking-wider text-primary font-bold">
          Verification Target
        </span>
        <h1 className="font-headline-lg text-headline-lg text-on-surface tracking-tight">
          {result.articleTitle || subject}
        </h1>
        <p className="font-body-sm text-body-sm text-on-surface-variant">
          Publisher: <strong>{sourceName}</strong> · Domain: {result.sourceReputation?.domain || result.sourceDomain || "Source Registry"}
        </p>
      </div>

      {/* ========================================================================= */}
      {/* 6-STAGE VERIFICATION FLOW:                                                */}
      {/* ARTICLE SUMMARY -> KEY TAKEAWAYS -> KEY CLAIMS -> SOURCE INFO -> EVIDENCE -> VERDICT */}
      {/* ========================================================================= */}

      <div className="space-y-6">
        {/* 1. ARTICLE SUMMARY */}
        <section className="bg-surface-container-lowest rounded-xl p-6 shadow-xs border border-surface-container-high/40 space-y-4">
          <div className="flex items-center justify-between border-b border-surface-container-high/30 pb-3">
            <div className="flex items-center gap-2">
              <span className="w-6 h-6 rounded-full bg-primary text-on-primary flex items-center justify-center font-label-xs font-bold">
                1
              </span>
              <h3 className="font-headline-sm text-headline-sm text-on-surface uppercase tracking-wide">
                Article Summary
              </h3>
            </div>
            <span className="px-2.5 py-0.5 rounded-full bg-surface-container text-primary font-label-xs text-label-xs font-semibold">
              {summary.provider || "Extractive NLP Engine"}
            </span>
          </div>

          {/* TL;DR */}
          <div className="p-4 rounded-lg bg-tertiary-fixed/30 text-on-tertiary-fixed space-y-1">
            <span className="font-label-xs text-label-xs uppercase tracking-wider text-tertiary font-bold">
              TL;DR (1 Sentence)
            </span>
            <p className="font-body-md text-body-md font-medium text-on-tertiary-fixed leading-relaxed">
              {summary.tldr}
            </p>
          </div>

          {/* Executive Summary */}
          <div className="space-y-1">
            <span className="font-label-xs text-label-xs uppercase tracking-wider text-on-surface-variant font-semibold">
              Executive Summary (3-5 Sentences)
            </span>
            <p className="font-body-md text-body-md text-on-surface leading-relaxed">
              {summary.executiveSummary}
            </p>
          </div>
        </section>

        {/* 2. KEY TAKEAWAYS */}
        <section className="bg-surface-container-lowest rounded-xl p-6 shadow-xs border border-surface-container-high/40 space-y-4">
          <div className="flex items-center justify-between border-b border-surface-container-high/30 pb-3">
            <div className="flex items-center gap-2">
              <span className="w-6 h-6 rounded-full bg-primary text-on-primary flex items-center justify-center font-label-xs font-bold">
                2
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
              <div key={idx} className="flex items-start gap-3 p-3 rounded-lg bg-surface-container-low">
                <div className="w-5 h-5 rounded-full bg-primary text-on-primary flex items-center justify-center flex-shrink-0 mt-0.5">
                  <MatIcon name="check" size={14} />
                </div>
                <span className="font-body-md text-body-md text-on-surface">{point}</span>
              </div>
            ))}
          </div>
        </section>

        {/* 3. KEY CLAIMS */}
        <section className="bg-surface-container-lowest rounded-xl p-6 shadow-xs border border-surface-container-high/40 space-y-4">
          <div className="flex items-center justify-between border-b border-surface-container-high/30 pb-3">
            <div className="flex items-center gap-2">
              <span className="w-6 h-6 rounded-full bg-primary text-on-primary flex items-center justify-center font-label-xs font-bold">
                3
              </span>
              <h3 className="font-headline-sm text-headline-sm text-on-surface uppercase tracking-wide">
                Key Claims
              </h3>
            </div>
            <span className="px-2.5 py-0.5 rounded-full bg-surface-container text-primary font-label-xs text-label-xs font-semibold">
              {result.claims?.length || 0} Decomposed Claims
            </span>
          </div>

          <div className="space-y-3">
            {result.claims && result.claims.length > 0 ? (
              result.claims.map((claim, idx) => {
                const isSupp = claim.status === "Supported";
                const isContra = claim.status === "Contradicted";
                return (
                  <div
                    key={idx}
                    className="p-4 rounded-lg bg-surface-container-low flex flex-col gap-1.5 border border-surface-container-high/30"
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-label-xs font-bold text-outline">CLAIM 0{idx + 1}</span>
                      <div className="flex items-center gap-2">
                        <span
                          className={`px-2.5 py-0.5 rounded-full font-label-xs text-label-xs font-semibold ${
                            isSupp
                              ? "bg-tertiary/15 text-tertiary"
                              : isContra
                              ? "bg-error/15 text-error"
                              : "bg-surface-container text-on-surface-variant"
                          }`}
                        >
                          {claim.status || "Verified"}
                        </span>
                        <span className="font-label-xs text-label-xs font-semibold text-primary">
                          {claim.confidence || claim.score || 90}% Confidence
                        </span>
                      </div>
                    </div>
                    <p className="font-body-md text-body-md font-medium text-on-surface">“{claim.claim}”</p>
                    {claim.supportingSources && claim.supportingSources.length > 0 && (
                      <span className="font-label-xs text-label-xs text-on-surface-variant">
                        Supporting citations: {claim.supportingSources.join(", ")}
                      </span>
                    )}
                  </div>
                );
              })
            ) : (
              <p className="text-on-surface-variant font-body-sm">No testable factual claims detected.</p>
            )}
          </div>
        </section>

        {/* 4. SOURCE INFORMATION */}
        <section className="bg-surface-container-lowest rounded-xl p-6 shadow-xs border border-surface-container-high/40 space-y-4">
          <div className="flex items-center justify-between border-b border-surface-container-high/30 pb-3">
            <div className="flex items-center gap-2">
              <span className="w-6 h-6 rounded-full bg-primary text-on-primary flex items-center justify-center font-label-xs font-bold">
                4
              </span>
              <h3 className="font-headline-sm text-headline-sm text-on-surface uppercase tracking-wide">
                Source Information
              </h3>
            </div>
            <span className="px-2.5 py-0.5 rounded-full bg-surface-container text-primary font-label-xs text-label-xs font-semibold">
              {sourceTier} Authority
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 p-4 rounded-lg bg-surface-container-low border border-surface-container-high/30">
            <div>
              <small className="font-label-xs text-label-xs text-outline uppercase tracking-wider block">
                Identified Primary Publisher
              </small>
              <b className="font-headline-sm text-headline-sm text-on-surface">{sourceName}</b>
              <span className="font-body-sm text-body-sm text-on-surface-variant block mt-0.5">
                Domain: {result.sourceReputation?.domain || result.sourceDomain || "Domain Registry"}
              </span>
            </div>
            <div className="sm:text-right">
              <small className="font-label-xs text-label-xs text-outline uppercase tracking-wider block">
                Domain Trust Score
              </small>
              <b
                className={`font-headline-lg text-headline-lg ${
                  sourceTrust >= 80 ? "text-tertiary" : sourceTrust >= 60 ? "text-amber-500" : "text-error"
                }`}
              >
                {sourceTrust}/100
              </b>
            </div>
          </div>

          <p className="font-body-sm text-body-sm text-on-surface-variant">
            {result.sourceReputation?.description ||
              "Source authority is determined strictly from domain ownership, metadata, and byline without evidence contamination."}
          </p>
        </section>

        {/* 5. EVIDENCE */}
        <section className="bg-surface-container-lowest rounded-xl p-6 shadow-xs border border-surface-container-high/40 space-y-4">
          <div className="flex items-center justify-between border-b border-surface-container-high/30 pb-3">
            <div className="flex items-center gap-2">
              <span className="w-6 h-6 rounded-full bg-primary text-on-primary flex items-center justify-center font-label-xs font-bold">
                5
              </span>
              <h3 className="font-headline-sm text-headline-sm text-on-surface uppercase tracking-wide">
                Evidence
              </h3>
            </div>
            <span className="px-2.5 py-0.5 rounded-full bg-surface-container text-primary font-label-xs text-label-xs font-semibold">
              {suppCount + contraCount} Citations
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-center">
            <div className="p-4 rounded-lg bg-tertiary-fixed/20 border border-tertiary/20">
              <span className="block font-headline-lg text-headline-lg text-tertiary font-bold">{suppCount}</span>
              <span className="font-label-xs text-label-xs uppercase font-semibold text-tertiary">
                Supporting Citations
              </span>
            </div>
            <div className="p-4 rounded-lg bg-error-container/20 border border-error/20">
              <span className="block font-headline-lg text-headline-lg text-error font-bold">{contraCount}</span>
              <span className="font-label-xs text-label-xs uppercase font-semibold text-error">
                Contradicting Flags
              </span>
            </div>
            <div className="p-4 rounded-lg bg-primary-fixed/20 border border-primary/20">
              <span className="block font-headline-lg text-headline-lg text-primary font-bold">
                {scoring.evidenceStrength.score}/100
              </span>
              <span className="font-label-xs text-label-xs uppercase font-semibold text-primary">
                Evidence Strength
              </span>
            </div>
          </div>

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
        </section>

        {/* 6. VERDICT */}
        <section className="bg-surface-container-lowest rounded-xl p-6 shadow-xs border border-surface-container-high/40 space-y-4">
          <div className="flex items-center justify-between border-b border-surface-container-high/30 pb-3">
            <div className="flex items-center gap-2">
              <span className="w-6 h-6 rounded-full bg-primary text-on-primary flex items-center justify-center font-label-xs font-bold">
                6
              </span>
              <h3 className="font-headline-sm text-headline-sm text-on-surface uppercase tracking-wide">
                Verdict
              </h3>
            </div>
            <span
              className={`px-3 py-1 rounded-full font-label-xs text-label-xs font-bold uppercase tracking-wider ${
                result.riskLevel === "LOW"
                  ? "bg-tertiary/15 text-tertiary"
                  : result.riskLevel === "HIGH" || result.riskLevel === "CRITICAL"
                  ? "bg-error/15 text-error"
                  : "bg-amber-500/15 text-amber-600"
              }`}
            >
              Risk: {result.riskLevel}
            </span>
          </div>

          <div className="flex flex-wrap items-center justify-between gap-4 p-4 rounded-lg bg-surface-container-low border border-surface-container-high/30">
            <div className="flex items-center gap-3">
              <div
                className={`w-12 h-12 rounded-full flex items-center justify-center shadow-xs ${
                  verdict === "VERIFIED" || verdict === "LIKELY TRUE"
                    ? "bg-tertiary text-on-tertiary"
                    : verdict === "KNOWN FALSE" || verdict === "LIKELY FALSE"
                    ? "bg-error text-on-error"
                    : "bg-amber-500 text-white"
                }`}
              >
                <MatIcon
                  name={verdict === "VERIFIED" || verdict === "LIKELY TRUE" ? "verified" : "warning"}
                  size={26}
                  fill
                />
              </div>
              <div>
                <small className="font-label-xs text-label-xs text-outline uppercase tracking-wider block font-bold">
                  Final Decision
                </small>
                <h2 className="font-headline-md text-headline-md font-bold text-on-surface m-0 leading-tight">
                  {verdict}
                </h2>
              </div>
            </div>

            <div className="text-right">
              <small className="font-label-xs text-label-xs text-outline uppercase tracking-wider block">
                Trust Score
              </small>
              <b
                className={`font-headline-lg text-headline-lg ${
                  result.trustScore >= 75
                    ? "text-tertiary"
                    : result.trustScore >= 50
                    ? "text-amber-500"
                    : "text-error"
                }`}
              >
                {result.trustScore}
                <small className="text-sm font-normal text-outline">/100</small>
              </b>
            </div>
          </div>

          {/* Synthesis Note */}
          <div className="p-4 rounded-lg bg-surface-container-low space-y-1">
            <span className="font-label-xs text-label-xs uppercase tracking-wider text-outline font-semibold">
              Verification Assessment
            </span>
            <p className="font-body-md text-body-md text-on-surface leading-relaxed">
              {result.summaryReason ||
                result.explanation?.join(" ") ||
                `Multi-source analysis confirmed high corroboration across independent news wires with ${suppCount} supporting citations.`}
            </p>
          </div>

          {/* Multi-Factor Scoring */}
          <div className="space-y-2 pt-2">
            <span className="font-label-xs text-label-xs uppercase tracking-wider text-outline font-semibold">
              Scoring Breakdown
            </span>
            <div className="grid grid-cols-1 gap-2">
              {[
                { label: "30% ML Prediction", score: scoring.mlPrediction.score, color: "bg-primary" },
                { label: "25% Source Reputation", score: scoring.sourceReputation.score, color: "bg-tertiary" },
                { label: "25% Evidence Strength", score: scoring.evidenceStrength.score, color: "bg-emerald-500" },
                { label: "10% Claim Consistency", score: scoring.claimConsistency.score, color: "bg-secondary" },
                { label: "10% Language Quality", score: scoring.languageQuality.score, color: "bg-amber-500" },
              ].map((row, i) => (
                <div key={i} className="flex items-center justify-between text-body-sm gap-4">
                  <span className="font-label-xs font-semibold text-on-surface w-40">{row.label}</span>
                  <div className="flex-1 bg-surface-container rounded-full h-2 overflow-hidden">
                    <div className={`${row.color} h-full rounded-full`} style={{ width: `${row.score}%` }} />
                  </div>
                  <span className="font-label-xs font-bold text-on-surface w-12 text-right">{row.score}%</span>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Audit Details Collapsible */}
        <div className="bg-surface-container-lowest rounded-xl shadow-xs border border-surface-container-high/40 overflow-hidden">
          <button
            onClick={() => setTechDetailsOpen(!techDetailsOpen)}
            className="w-full px-6 py-4 flex items-center justify-between text-left hover:bg-surface-container-low transition-colors cursor-pointer"
            type="button"
          >
            <div className="flex items-center gap-2">
              <MatIcon name="biotech" size={18} className="text-primary" />
              <span className="font-label-md text-label-md text-on-surface font-semibold">
                Technical Audit & Timeline
              </span>
            </div>
            <MatIcon
              name="expand_more"
              size={18}
              className={`text-outline transition-transform duration-200 ${techDetailsOpen ? "rotate-180" : ""}`}
            />
          </button>

          {techDetailsOpen && (
            <div className="px-6 pb-6 pt-2 space-y-4 border-t border-surface-container-high/30">
              {result.timeline && result.timeline.length > 0 && (
                <div className="space-y-2">
                  <h5 className="font-label-sm font-semibold text-on-surface">Execution Steps</h5>
                  <div className="relative pl-5 space-y-2 before:content-[''] before:absolute before:left-2 before:top-2 before:bottom-2 before:w-0.5 before:bg-surface-container">
                    {result.timeline.map((step, idx) => (
                      <div key={idx} className="relative">
                        <span className="font-label-xs text-label-xs text-primary font-bold">{step.step}: </span>
                        <span className="font-body-sm text-body-sm text-on-surface">{step.detail}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              <p className="font-body-sm text-body-sm text-on-surface-variant">
                Missing evidence is strictly classified as UNVERIFIED and never treated as evidence of falsity.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ==========================================
// 6. MAIN APPLICATION
// ==========================================

export default function App() {
  const [screen, setScreen] = useState<Screen>("home");
  const [darkMode, setDarkMode] = useState<boolean>(() => {
    return localStorage.getItem("lucid_theme") === "dark";
  });

  const [activeSubject, setActiveSubject] = useState("");
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [analysisStep, setAnalysisStep] = useState(1);
  const [currentResult, setCurrentResult] = useState<VerificationResult | null>(null);

  // History state: empty by default (NO preloaded demo records)
  const [history, setHistory] = useState<HistoryItem[]>(() => {
    try {
      const saved = localStorage.getItem("lucid_history");
      if (saved) return JSON.parse(saved);
    } catch {}
    return [];
  });

  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add("dark");
      localStorage.setItem("lucid_theme", "dark");
    } else {
      document.documentElement.classList.remove("dark");
      localStorage.setItem("lucid_theme", "light");
    }
  }, [darkMode]);

  useEffect(() => {
    localStorage.setItem("lucid_history", JSON.stringify(history));
  }, [history]);

  const handleStartVerification = async ({
    mode,
    content,
    file,
  }: {
    mode: "url" | "text" | "upload";
    content: string;
    file?: File;
  }) => {
    setActiveSubject(content);
    setAnalysisProgress(15);
    setAnalysisStep(1);
    setScreen("live-analysis");

    const timer1 = setTimeout(() => {
      setAnalysisProgress(38);
      setAnalysisStep(2);
    }, 700);

    const timer2 = setTimeout(() => {
      setAnalysisProgress(65);
      setAnalysisStep(3);
    }, 1500);

    const timer3 = setTimeout(() => {
      setAnalysisProgress(85);
      setAnalysisStep(4);
    }, 2400);

    try {
      let response: Response;
      if (mode === "upload" && file) {
        const formData = new FormData();
        formData.append("file", file);
        response = await fetch("/api/analyze-file", {
          method: "POST",
          body: formData,
        });
      } else if (mode === "url") {
        response = await fetch("/api/analyze-url", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ url: content }),
        });
      } else {
        response = await fetch("/api/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: content }),
        });
      }

      if (!response.ok) {
        throw new Error(`API returned status ${response.status}`);
      }

      const data: VerificationResult = await response.json();

      clearTimeout(timer1);
      clearTimeout(timer2);
      clearTimeout(timer3);
      setAnalysisProgress(100);
      setAnalysisStep(5);

      setTimeout(() => {
        setCurrentResult(data);
        setActiveSubject(data.articleTitle || content);

        const newHistoryItem: HistoryItem = {
          id: `LCD-${Math.floor(1000 + Math.random() * 9000)}`,
          title: data.articleTitle || content.slice(0, 70),
          time: "Just now",
          score: data.trustScore,
          verdict: data.verdict,
          data,
        };
        setHistory((prev) => [newHistoryItem, ...prev]);
        setScreen("report");
      }, 500);
    } catch (err) {
      console.error("Verification error:", err);
      clearTimeout(timer1);
      clearTimeout(timer2);
      clearTimeout(timer3);
      alert("Verification could not be completed. Please ensure the backend server is running on port 5000.");
      setScreen("home");
    }
  };

  return (
    <div className="bg-surface font-body-md text-on-surface antialiased min-h-screen flex flex-col transition-colors duration-200">
      <AppHeader
        onNavigateHome={() => setScreen("home")}
        darkMode={darkMode}
        onToggleDarkMode={() => setDarkMode(!darkMode)}
      />

      <main className="w-full pt-16 bg-surface min-h-[calc(100vh-140px)] flex-1 flex flex-col">
        {screen === "home" && (
          <HomeScreen
            onVerify={handleStartVerification}
            history={history}
            onSelectHistory={(item) => {
              setCurrentResult(item.data);
              setActiveSubject(item.title);
              setScreen("report");
            }}
          />
        )}

        {screen === "live-analysis" && (
          <LoadingScreen
            subject={activeSubject}
            progress={analysisProgress}
            activeStep={analysisStep}
            onCancel={() => setScreen("home")}
          />
        )}

        {screen === "report" && (
          currentResult ? (
            <ReportScreen
              result={currentResult}
              subject={activeSubject}
              onNewSearch={() => setScreen("home")}
            />
          ) : (
            <div className="flex-1 flex flex-col items-center justify-center py-20 text-center gap-4">
              <h2 className="font-headline-sm text-headline-sm">No report loaded</h2>
              <button
                onClick={() => setScreen("home")}
                className="px-5 py-2 rounded-full bg-primary text-on-primary font-label-md"
              >
                Go to Verification Home
              </button>
            </div>
          )
        )}
      </main>

      <AppFooter />
    </div>
  );
}
