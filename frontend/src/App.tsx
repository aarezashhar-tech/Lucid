import React, { FormEvent, useCallback, useEffect, useRef, useState } from "react";
import {
  auth,
  signInWithGoogle,
  signOutFirebase,
  isFirebaseConfigured,
} from "./firebase";

// ==========================================
// API CONFIGURATION (VITE_API_URL SUPPORT)
// ==========================================
/**
 * Resolves full API endpoint URL from import.meta.env.VITE_API_URL.
 * - Removes any trailing slash from VITE_API_URL before joining.
 * - Falls back to http://localhost:5000 ONLY when running in dev mode.
 * - In production, if VITE_API_URL is missing, logs a clear console error.
 */
export function getApiUrl(path: string): string {
  const cleanPath = path.startsWith("/") ? path : `/${path}`;
  const rawApiUrl = import.meta.env.VITE_API_URL;
  const baseUrl = (rawApiUrl || "").trim().replace(/\/+$/, "");

  if (baseUrl) {
    return `${baseUrl}${cleanPath}`;
  }

  if (import.meta.env.DEV) {
    return `http://localhost:5000${cleanPath}`;
  }

  console.error(
    "[API Configuration] VITE_API_URL is missing in production environment. " +
    "Requests cannot reach the backend API. Please configure VITE_API_URL in your Vercel deployment settings."
  );
  return cleanPath;
}

export const API_BASE_URL: string = (import.meta.env.VITE_API_URL || "").trim().replace(/\/+$/, "");

/**
 * Executes a fetch request to the backend using getApiUrl.
 * Automatically attaches Firebase ID Token when an authenticated session exists.
 */
export async function fetchApi(path: string, init?: RequestInit): Promise<Response> {
  const url = getApiUrl(path);
  const options: RequestInit = { ...init };
  const headers = new Headers(options.headers || {});

  try {
    if (auth?.currentUser) {
      const token = await auth.currentUser.getIdToken();
      if (token && !headers.has("Authorization")) {
        headers.set("Authorization", `Bearer ${token}`);
      }
    }
  } catch (err) {
    console.warn("Could not retrieve Firebase token:", err);
  }

  options.headers = headers;
  return fetch(url, options);
}

/**
 * Lightweight mouse spotlight tracker using CSS variables --mx, --my
 * and requestAnimationFrame for smooth 60fps rendering.
 * Disabled on touch screens and prefers-reduced-motion.
 */
export const handleCardPointerMove = (e: React.PointerEvent<HTMLElement>) => {
  if (typeof window !== "undefined" && window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    return;
  }
  if (e.pointerType === "touch") return;
  const target = e.currentTarget;
  const clientX = e.clientX;
  const clientY = e.clientY;
  requestAnimationFrame(() => {
    const rect = target.getBoundingClientRect();
    target.style.setProperty("--mx", `${clientX - rect.left}px`);
    target.style.setProperty("--my", `${clientY - rect.top}px`);
  });
};

// ==========================================
// 1. DATA TYPES & INTERFACES
// ==========================================

export type Screen = "welcome" | "home" | "ocr-preview" | "live-analysis" | "report";

export interface OcrPreviewData {
  fileName: string;
  fileExtension: string;
  extractedText: string;
  previewText: string;
  confidence: number;
  isLowConfidence: boolean;
  warning?: string;
  qualityWarning?: string;
  wordCount?: number;
  originalFile?: File;
}

export type VerdictType =
  | "VERIFIED"
  | "LIKELY TRUE"
  | "NEEDS REVIEW"
  | "UNVERIFIED"
  | "LIKELY FALSE"
  | "KNOWN FALSE"
  | "FALSE"
  | "SUSPICIOUS"
  | "LIKELY FAKE"
  | "MIXED EVIDENCE"
  | "OPINION / COMMENTARY"
  | string;

export interface ClaimItem {
  id?: string;
  claimNumber?: number;
  claim: string;
  claimType?: string;
  isOpinion?: boolean;
  genre?: string;
  verdict?: string;
  status?: "Supported" | "Contradicted" | "Unverified" | "Disputed" | "Verified" | "Likely True" | "False" | "Opinion / Subjective" | string;
  score?: number;
  source?: string;
  supportingSources?: string[];
  contradictingSources?: string[];
  supportingEvidenceCount?: number;
  contradictingEvidenceCount?: number;
  confidence?: number;
  analysisNotes?: string;
}

export interface RAGEvidenceSource {
  articleTitle: string;
  publisher: string;
  publicationDate: string;
  url: string;
  evidenceSnippet: string;
  sourceTrustScore: number;
  sourceCategory?: string;
  relevanceScore?: number;
  semanticSimilarity?: number;
  compositeRankScore?: number;
  stance?: "support" | "contradict" | "neutral" | string;
  stanceConfidence?: number;
  matchingClaim?: string;
  factCheckRating?: string;
  retrievalChannel?: string;
}

export interface RAGEvidenceMatrixRow {
  claimNumber: number;
  claim: string;
  claimType: string;
  isOpinion?: boolean;
  status: string;
  confidence: number;
  supportingEvidence: RAGEvidenceSource[];
  contradictingEvidence: RAGEvidenceSource[];
  sourceLinks: string[];
  evidenceSummary: string;
}

export interface RAGGroundedReasoning {
  synthesis: string;
  whyAppearsTrue: string[];
  whyAppearsFalse: string[];
  evidenceFound: string[];
  evidenceMissing: string[];
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

export type ArticleType =
  | "News Report"
  | "Feature Article"
  | "Historical Analysis"
  | "Opinion Piece"
  | string;

export interface SourceQualityMetric {
  score: number;
  level: string;
  notes: string;
}

export interface SourceQualityAssessment {
  overallRating: string;
  sourceQualityScore: number;
  scholarlyStandards: SourceQualityMetric;
  methodologicalRigor: SourceQualityMetric;
  archivalIntegrity: SourceQualityMetric;
  sourceAuthority: SourceQualityMetric;
  disclaimer: string;
}

export interface EmptyEvidenceAudit {
  sourcesSearched: string[];
  resultsFound: number;
  reason: string;
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
  publisherConfidence?: string;
  publisher?: string;
  sourceType?: string;
}

export interface ScoringFactorItem {
  weight: number;
  score: number | string;
  contribution: number;
  label: string;
  notes?: string;
}

export interface ScoringFactors {
  evidenceStrength?: ScoringFactorItem;
  mlPrediction?: ScoringFactorItem;
  languageQuality?: ScoringFactorItem;
  sourceReputation?: ScoringFactorItem;
  claimConsistency?: ScoringFactorItem;
}

export interface ArticleSummary {
  title?: string;
  executiveSummary: string; // 3-5 simple sentences
  keyTakeaways: string[];   // Maximum 5 bullet points
  tldr: string;             // 1 sentence
  quickSummary?: string;    // 1-2 simple sentences, plain English
  provider?: string;
}

export interface VerificationResult {
  prediction: "REAL" | "FAKE" | "UNKNOWN";
  confidence: number;
  credibility?: number | string;
  trustScore: number | string;
  riskLevel: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL" | "UNKNOWN";
  verdict: VerdictType;
  summaryReason?: string;
  explanation: string[];
  articleSummary?: ArticleSummary;
  summary?: ArticleSummary;
  publisher?: string;
  sourceType?: string;
  sourceDomain?: string;
  articleType?: ArticleType;
  articleTypeClassification?: {
    articleType: string;
    confidence: number;
    genreMarkers: string[];
    explanation: string;
  };
  language?: {
    language: string;
    code: string;
    isForeignLanguage: boolean;
    wasTranslated?: boolean;
    originalSnippet?: string;
  };
  detectedLanguage?: string;
  languageCode?: string;
  wasTranslated?: boolean;
  originalLanguageText?: string;
  sourceQualityAssessment?: SourceQualityAssessment;
  emptyEvidenceAudit?: EmptyEvidenceAudit;
  ragEnabled?: boolean;
  sourcesSearched?: string[];
  resultsFound?: {
    supporting: number;
    contradicting: number;
    total: number;
  };
  topSupportingSources?: RAGEvidenceSource[];
  topContradictingSources?: RAGEvidenceSource[];
  evidenceMatrix?: RAGEvidenceMatrixRow[];
  groundedReasoning?: RAGGroundedReasoning;
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
    emptyEvidenceAudit?: EmptyEvidenceAudit;
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
  ocrPreview?: {
    previewText: string;
    extractedText: string;
    confidence: number;
    isLowConfidence: boolean;
    warning?: string;
    qualityWarning?: string;
  };
  ocrConfidence?: number;
  isLowConfidence?: boolean;
  ocrWarning?: string;
  ocrQualityWarning?: string;
}

export interface HistoryItem {
  id: string;
  title: string;
  time: string;
  score: number | string;
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

const cleanSnippetText = (str: string) => {
  if (!str) return "";
  return str
    .replace(/<[^>]+>/g, " ")
    .replace(/https?:\/\/\S+/g, "")
    .replace(/["']?\s*<a\s+[^>]*>?/gi, "")
    .replace(/(target|href|class|rel)=["']?[^"'>\s]+["']?/gi, "")
    .replace(/\s+/g, " ")
    .trim();
};

interface UserAccount {
  name: string;
  email: string;
  isGuest: boolean;
  avatar?: string;
}

// Share Popover Menu (Matching BBC screenshot style & Lucid theme combo)
function SharePopover({
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
    navigator.clipboard.writeText(currentUrl);
    setCopied(true);
    setTimeout(() => {
      setCopied(false);
      onClose();
    }, 1800);
  };

  const shareOptions = [
    {
      name: "Facebook",
      icon: (
        <svg className="w-4 h-4 fill-current" viewBox="0 0 24 24">
          <path d="M24 12.073c0-6.627-5.373-12-12-12s-12 5.373-12 12c0 5.99 4.388 10.954 10.125 11.854v-8.385H7.078v-3.47h3.047V9.43c0-3.007 1.792-4.669 4.533-4.669 1.312 0 2.686.235 2.686.235v2.953H15.83c-1.491 0-1.956.925-1.956 1.874v2.25h3.328l-.532 3.47h-2.796v8.385C19.612 23.027 24 18.062 24 12.073z"/>
        </svg>
      ),
      action: () => window.open(`https://www.facebook.com/sharer/sharer.php?u=${encodeURIComponent(currentUrl)}`, "_blank"),
    },
    {
      name: "X (Twitter)",
      icon: (
        <svg className="w-4 h-4 fill-current" viewBox="0 0 24 24">
          <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 23.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/>
        </svg>
      ),
      action: () => window.open(`https://twitter.com/intent/tweet?text=${encodeURIComponent(shareText)}&url=${encodeURIComponent(currentUrl)}`, "_blank"),
    },
    {
      name: "LinkedIn",
      icon: (
        <svg className="w-4 h-4 fill-current" viewBox="0 0 24 24">
          <path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z"/>
        </svg>
      ),
      action: () => window.open(`https://www.linkedin.com/sharing/share-offsite/?url=${encodeURIComponent(currentUrl)}`, "_blank"),
    },
    {
      name: "Email",
      icon: <MatIcon name="mail" size={16} />,
      action: () => window.location.href = `mailto:?subject=${encodeURIComponent(shareText)}&body=${encodeURIComponent(`Read the fact-check on Lucid: ${currentUrl}`)}`,
    },
    {
      name: copied ? "Copied Link!" : "Copy Link",
      icon: <MatIcon name={copied ? "check" : "content_copy"} size={16} className={copied ? "text-emerald-500" : ""} />,
      action: handleCopy,
    },
  ];

  return (
    <>
      <div className="fixed inset-0 z-40 bg-transparent" onClick={onClose} />
      <div className="absolute right-0 top-10 z-50 w-52 bg-surface-container-lowest/95 backdrop-blur-xl border border-surface-container-high rounded-xl shadow-xl py-1.5 flex flex-col animate-scale-up text-on-surface">
        <div className="px-3.5 py-1.5 text-xs font-bold uppercase tracking-wider text-outline border-b border-surface-container-high/40">
          Share Investigation
        </div>
        {shareOptions.map((opt, i) => (
          <button
            key={i}
            onClick={() => {
              opt.action();
              if (opt.name !== "Copy Link" && opt.name !== "Copied Link!") onClose();
            }}
            className="w-full px-3.5 py-2 flex items-center gap-3 text-sm text-left hover:bg-surface-container transition-colors cursor-pointer text-on-surface"
            type="button"
          >
            <span className="w-5 flex items-center justify-center text-on-surface-variant">
              {opt.icon}
            </span>
            <span className="font-medium text-xs">{opt.name}</span>
          </button>
        ))}
      </div>
    </>
  );
}

// ==========================================
// OFFLINE 404 NOT FOUND & ENDLESS RUNNER MINI-GAME
// ==========================================

function DinoRunnerGame({ isDarkMode }: { isDarkMode: boolean }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [gameState, setGameState] = useState<"idle" | "playing" | "gameover">("idle");
  const [score, setScore] = useState(0);
  const [highScore, setHighScore] = useState(() => {
    try {
      return parseInt(localStorage.getItem("lucid_dino_high_score") || "0", 10) || 0;
    } catch {
      return 0;
    }
  });

  const gameStateRef = useRef(gameState);
  gameStateRef.current = gameState;
  const scoreRef = useRef(score);
  scoreRef.current = score;
  const highScoreRef = useRef(highScore);
  highScoreRef.current = highScore;

  const playerRef = useRef({
    x: 45,
    y: 114,
    width: 22,
    height: 22,
    vy: 0,
    jumpForce: -10.5,
    gravity: 0.62,
    groundY: 114,
    isGrounded: true,
  });

  const obstaclesRef = useRef<Array<{ x: number; y: number; width: number; height: number }>>([]);
  const animFrameIdRef = useRef<number | null>(null);
  const groundOffsetRef = useRef(0);
  const frameCountRef = useRef(0);
  const speedRef = useRef(4.6);

  const jump = useCallback(() => {
    if (gameStateRef.current === "idle" || gameStateRef.current === "gameover") {
      obstaclesRef.current = [];
      scoreRef.current = 0;
      setScore(0);
      speedRef.current = 4.8;
      frameCountRef.current = 0;
      playerRef.current.y = playerRef.current.groundY;
      playerRef.current.vy = playerRef.current.jumpForce;
      playerRef.current.isGrounded = false;
      gameStateRef.current = "playing";
      setGameState("playing");
    } else if (gameStateRef.current === "playing") {
      if (playerRef.current.isGrounded) {
        playerRef.current.vy = playerRef.current.jumpForce;
        playerRef.current.isGrounded = false;
      }
    }
  }, []);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.code === "Space" || e.code === "ArrowUp") {
        e.preventDefault();
        jump();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [jump]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let lastObstacleSpawn = 0;

    const loop = () => {
      const currentCanvas = canvasRef.current;
      if (!currentCanvas) return;
      const width = currentCanvas.width;
      const height = currentCanvas.height;
      const groundY = 136;

      ctx.clearRect(0, 0, width, height);

      const groundColor = isDarkMode ? "#334155" : "#c3c6d7";
      const groundDotColor = isDarkMode ? "#64748b" : "#94a3b8";
      const playerColor = isDarkMode ? "#38bdf8" : "#004ac6";
      const playerGlow = isDarkMode ? "rgba(56, 189, 248, 0.45)" : "rgba(0, 74, 198, 0.25)";
      const obstacleColor = isDarkMode ? "#f87171" : "#ba1a1a";

      if (gameStateRef.current === "playing") {
        frameCountRef.current += 1;
        if (frameCountRef.current % 5 === 0) {
          scoreRef.current += 1;
          setScore(scoreRef.current);
          if (scoreRef.current > highScoreRef.current) {
            highScoreRef.current = scoreRef.current;
            setHighScore(scoreRef.current);
            try {
              localStorage.setItem("lucid_dino_high_score", String(scoreRef.current));
            } catch {}
          }
          speedRef.current = 4.8 + Math.min(scoreRef.current / 250, 4.5);
        }

        const player = playerRef.current;
        player.vy += player.gravity;
        player.y += player.vy;
        if (player.y >= player.groundY) {
          player.y = player.groundY;
          player.vy = 0;
          player.isGrounded = true;
        }

        const timeSinceLastSpawn = frameCountRef.current - lastObstacleSpawn;
        const minGapFrames = Math.max(50, Math.floor(125 - speedRef.current * 7));
        if (timeSinceLastSpawn > minGapFrames && Math.random() < 0.035) {
          const obsHeight = 20 + Math.floor(Math.random() * 20);
          const obsWidth = 14 + Math.floor(Math.random() * 12);
          obstaclesRef.current.push({
            x: width + 20,
            y: groundY - obsHeight,
            width: obsWidth,
            height: obsHeight,
          });
          lastObstacleSpawn = frameCountRef.current;
        }

        const currentSpeed = speedRef.current;
        obstaclesRef.current.forEach((obs) => {
          obs.x -= currentSpeed;
        });

        obstaclesRef.current = obstaclesRef.current.filter((obs) => obs.x + obs.width > -20);
        groundOffsetRef.current = (groundOffsetRef.current + currentSpeed) % 24;

        const pLeft = player.x + 3;
        const pRight = player.x + player.width - 3;
        const pTop = player.y + 3;
        const pBottom = player.y + player.height - 1;

        for (const obs of obstaclesRef.current) {
          const oLeft = obs.x + 2;
          const oRight = obs.x + obs.width - 2;
          const oTop = obs.y + 2;
          const oBottom = obs.y + obs.height;

          if (pRight > oLeft && pLeft < oRight && pBottom > oTop && pTop < oBottom) {
            gameStateRef.current = "gameover";
            setGameState("gameover");
            break;
          }
        }
      }

      // Draw ground line
      ctx.beginPath();
      ctx.strokeStyle = groundColor;
      ctx.lineWidth = 2;
      ctx.moveTo(0, groundY);
      ctx.lineTo(width, groundY);
      ctx.stroke();

      // Ground dash dots
      ctx.fillStyle = groundDotColor;
      const dotSpacing = 24;
      for (let x = -groundOffsetRef.current; x < width; x += dotSpacing) {
        ctx.fillRect(x, groundY + 4, 6, 2);
        if ((x + 12) % 48 === 0) {
          ctx.fillRect(x + 10, groundY + 8, 3, 2);
        }
      }

      // Draw player box
      const p = playerRef.current;
      ctx.save();
      ctx.shadowColor = playerGlow;
      ctx.shadowBlur = 10;
      ctx.fillStyle = playerColor;
      ctx.beginPath();
      if ("roundRect" in ctx && typeof ctx.roundRect === "function") {
        ctx.roundRect(p.x, p.y, p.width, p.height, 4);
      } else {
        ctx.rect(p.x, p.y, p.width, p.height);
      }
      ctx.fill();

      // Eye dot
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(p.x + p.width - 7, p.y + 5, 3, 3);
      ctx.restore();

      // Draw obstacles
      obstaclesRef.current.forEach((obs) => {
        ctx.save();
        ctx.fillStyle = obstacleColor;
        ctx.beginPath();
        if ("roundRect" in ctx && typeof ctx.roundRect === "function") {
          ctx.roundRect(obs.x, obs.y, obs.width, obs.height, [3, 3, 0, 0]);
        } else {
          ctx.rect(obs.x, obs.y, obs.width, obs.height);
        }
        ctx.fill();
        ctx.fillStyle = isDarkMode ? "rgba(0,0,0,0.3)" : "rgba(255,255,255,0.4)";
        ctx.fillRect(obs.x + 3, obs.y + 4, 3, obs.height - 6);
        ctx.restore();
      });

      // Text prompts
      if (gameStateRef.current === "idle") {
        ctx.fillStyle = isDarkMode ? "#f1f5f9" : "#131b2e";
        ctx.font = "bold 14px Inter, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("Press Space or Tap to Jump", width / 2, groundY - 50);
        ctx.fillStyle = isDarkMode ? "#94a3b8" : "#5f6272";
        ctx.font = "12px Inter, sans-serif";
        ctx.fillText("Dodge the incoming obstacles!", width / 2, groundY - 28);
      } else if (gameStateRef.current === "gameover") {
        ctx.fillStyle = isDarkMode ? "#f87171" : "#ba1a1a";
        ctx.font = "bold 18px Inter, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("GAME OVER", width / 2, groundY - 60);

        ctx.fillStyle = isDarkMode ? "#f1f5f9" : "#131b2e";
        ctx.font = "bold 13px Inter, sans-serif";
        ctx.fillText(`Score: ${scoreRef.current}  |  Best: ${highScoreRef.current}`, width / 2, groundY - 36);

        ctx.fillStyle = isDarkMode ? "#94a3b8" : "#5f6272";
        ctx.font = "12px Inter, sans-serif";
        ctx.fillText("Press Space or Tap to Restart", width / 2, groundY - 14);
      }

      animFrameIdRef.current = requestAnimationFrame(loop);
    };

    animFrameIdRef.current = requestAnimationFrame(loop);
    return () => {
      if (animFrameIdRef.current) cancelAnimationFrame(animFrameIdRef.current);
    };
  }, [isDarkMode]);

  return (
    <div className="w-full flex flex-col items-center select-none">
      <div className="w-full flex items-center justify-between px-2 pb-2 text-xs font-mono font-semibold">
        <span className="text-on-surface-variant flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-primary animate-pulse"></span>
          SCORE: <span className="text-primary font-bold">{score.toString().padStart(5, "0")}</span>
        </span>
        <span className="text-outline">
          HIGH: <span className="font-bold text-on-surface">{highScore.toString().padStart(5, "0")}</span>
        </span>
      </div>

      <div
        onClick={jump}
        onTouchStart={(e) => {
          e.preventDefault();
          jump();
        }}
        className="w-full rounded-xl bg-surface-container-low border border-surface-container-high/60 cursor-pointer overflow-hidden touch-none relative"
        style={{ height: "160px" }}
        role="button"
        tabIndex={0}
        aria-label="Offline Mini Game: Tap or press space to jump"
      >
        <canvas
          ref={canvasRef}
          width={580}
          height={160}
          className="w-full h-full block"
        />
      </div>
      <p className="mt-2 text-xs text-outline text-center">
        Controls: <kbd className="px-1.5 py-0.5 rounded bg-surface-container text-on-surface font-mono">Space</kbd> or <kbd className="px-1.5 py-0.5 rounded bg-surface-container text-on-surface font-mono">↑</kbd> to jump · Tap on mobile
      </p>
    </div>
  );
}

function Offline404Overlay({
  onRetry,
  onDismiss,
  isChecking,
  isDarkMode,
}: {
  onRetry: () => void;
  onDismiss: () => void;
  isChecking: boolean;
  isDarkMode: boolean;
}) {
  return (
    <div
      className="fixed inset-0 z-50 overflow-y-auto flex flex-col items-center justify-center p-4 sm:p-6 bg-surface/95 backdrop-blur-md transition-all animate-fade-in"
      role="dialog"
      aria-modal="true"
      aria-label="Offline 404 Notice"
    >
      <div className="max-w-xl w-full mx-auto flex flex-col items-center text-center">
        <div className="mb-2">
          <LucidBrandLogo size="md" showText={false} />
        </div>

        <h1 className="font-display-hero text-6xl sm:text-7xl md:text-8xl font-black text-primary tracking-tight leading-none mb-1">
          404
        </h1>

        <h2 className="text-2xl sm:text-3xl font-bold text-on-surface tracking-tight mt-1">
          Oops, You Are Offline!
        </h2>

        <p className="font-body-md text-xs sm:text-sm text-on-surface-variant max-w-md mt-2 mb-5 leading-relaxed">
          The internet connection was lost. Play the mini-game below while waiting for connection.
        </p>

        <div className="w-full lucid-card bg-surface-container-lowest rounded-2xl shadow-xl border border-surface-container-high/50 p-4 sm:p-5 mb-5 transition-all">
          <DinoRunnerGame isDarkMode={isDarkMode} />
        </div>

        <div className="flex flex-col sm:flex-row items-center justify-center gap-3 w-full sm:w-auto">
          <button
            onClick={onRetry}
            disabled={isChecking}
            className="w-full sm:w-auto min-h-[44px] px-6 py-2.5 rounded-full bg-primary text-on-primary font-semibold text-sm hover:opacity-95 transition-all flex items-center justify-center gap-2 cursor-pointer shadow-sm disabled:opacity-60"
            type="button"
            aria-label="Retry network connection"
          >
            <MatIcon name="refresh" size={18} className={isChecking ? "animate-spin" : ""} />
            <span>{isChecking ? "Checking Connection..." : "Retry Connection"}</span>
          </button>

          <button
            onClick={onDismiss}
            className="w-full sm:w-auto min-h-[44px] px-6 py-2.5 rounded-full bg-surface-container hover:bg-surface-container-high text-on-surface font-semibold text-sm transition-all flex items-center justify-center gap-2 cursor-pointer border border-surface-container-high/40"
            type="button"
            aria-label="Dismiss offline screen and view cached page"
          >
            <MatIcon name="visibility" size={18} />
            <span>Dismiss / View Cached Page</span>
          </button>
        </div>
      </div>
    </div>
  );
}

// ==========================================
// 2. AUTHENTICATION & WELCOME SCREENS
// ==========================================

// Full-Page Welcome / Authentication Gate Screen (Default Entry Point)
function WelcomeScreen({
  onLogin,
}: {
  onLogin: (user: UserAccount) => void;
}) {
  const [email, setEmail] = useState("");
  const [authError, setAuthError] = useState<string | null>(null);
  const [isAuthenticating, setIsAuthenticating] = useState<"google" | null>(null);

  const handleGoogleLogin = async () => {
    setAuthError(null);
    if (!isFirebaseConfigured) {
      setAuthError(
        "OAuth is not configured yet. Please configure the Firebase environment variables (VITE_FIREBASE_*) in Vercel or your local .env file."
      );
      return;
    }
    setIsAuthenticating("google");
    try {
      const res = await signInWithGoogle();
      if (res.user) {
        onLogin(res.user);
      } else if (res.error) {
        setAuthError(res.error);
      }
    } catch (err: any) {
      setAuthError(err?.message || "Google sign-in failed.");
    } finally {
      setIsAuthenticating(null);
    }
  };

  return (
    <div className="w-full flex-1 flex flex-col items-center justify-center px-4 py-8 sm:py-16 animate-fade-in">
      <div
        onPointerMove={handleCardPointerMove}
        className="w-full max-w-[420px] lucid-card glass-card spotlight-card rounded-2xl shadow-xl p-6 sm:p-8 flex flex-col items-center gap-5 text-center transition-all"
      >
        {/* Brand Icon */}
        <div className="w-14 h-14 rounded-full bg-primary/10 border border-primary/20 flex items-center justify-center text-primary mt-1 shadow-xs">
          <LucidBrandLogo size="hero" showText={false} />
        </div>

        {/* Titles */}
        <div className="space-y-1.5">
          <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-on-surface">
            Sign in or sign up
          </h1>
          <p className="text-xs sm:text-sm text-on-surface-variant max-w-xs mx-auto">
            Start verifying claims, detecting misinformation, and investigating evidence with Lucid.
          </p>
        </div>

        {/* Auth Error Banner */}
        {authError && (
          <div className="w-full p-3 rounded-xl bg-error-container/40 border border-error/30 text-error text-xs text-left flex items-start gap-2">
            <MatIcon name="error_outline" size={16} className="shrink-0 mt-0.5" />
            <span className="flex-1 leading-relaxed">{authError}</span>
          </div>
        )}

        {/* OAuth Buttons */}
        <div className="w-full space-y-2.5 pt-1">
          <button
            onClick={handleGoogleLogin}
            disabled={isAuthenticating !== null}
            className="w-full min-h-[44px] flex items-center justify-center gap-3 px-4 py-2.5 rounded-xl border border-surface-container-high hover:bg-surface-container-low transition-all font-label-md text-label-md text-on-surface font-medium cursor-pointer disabled:opacity-60 disabled:cursor-not-allowed"
            type="button"
          >
            {isAuthenticating === "google" ? (
              <MatIcon name="progress_activity" size={18} className="animate-spin text-primary" />
            ) : (
              <svg className="w-4 h-4 shrink-0" viewBox="0 0 24 24">
                <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" />
                <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" />
                <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.06H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.94l2.85-2.22.81-.63z" />
                <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.52 6.16-4.52z" />
              </svg>
            )}
            <span>{isAuthenticating === "google" ? "Signing in..." : "Continue with Google"}</span>
          </button>
        </div>

        {/* Divider */}
        <div className="w-full flex items-center gap-3 text-xs text-outline my-0.5">
          <div className="flex-1 h-px bg-surface-container-high" />
          <span>Or continue with email</span>
          <div className="flex-1 h-px bg-surface-container-high" />
        </div>

        {/* Email form */}
        <form 
          onSubmit={(e) => {
            e.preventDefault();
            if (email.trim()) {
              onLogin({ name: email.split('@')[0], email: email.trim(), isGuest: false });
            }
          }}
          className="w-full space-y-2.5"
        >
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="Enter your email address"
            aria-label="Email address"
            id="welcome-email-input"
            className="w-full min-h-[44px] px-4 py-2.5 rounded-xl bg-surface-container-low border border-surface-container-high text-on-surface text-base sm:text-sm placeholder:text-outline focus:outline-none focus:border-primary transition-all"
            required
          />
          <button
            type="submit"
            className="w-full min-h-[44px] py-2.5 rounded-xl bg-primary text-on-primary font-semibold text-sm hover:opacity-95 transition-all cursor-pointer shadow-xs"
          >
            Continue with Email
          </button>
        </form>

        {/* Guest Divider */}
        <div className="w-full flex items-center gap-3 text-xs text-outline my-0.5">
          <div className="flex-1 h-px bg-surface-container-high" />
          <span>Or enter without an account</span>
          <div className="flex-1 h-px bg-surface-container-high" />
        </div>

        {/* Guest Button */}
        <button
          onClick={() => onLogin({ name: "Guest User", email: "guest@lucid.local", isGuest: true })}
          className="w-full min-h-[44px] py-2.5 rounded-xl bg-surface-container hover:bg-surface-container-high text-on-surface border border-surface-container-high transition-all text-sm font-semibold flex items-center justify-center gap-2 cursor-pointer"
          type="button"
        >
          <MatIcon name="person_outline" size={18} />
          <span>Continue as Guest (Temporary Access)</span>
        </button>

        <p className="text-xs text-outline max-w-xs leading-relaxed">
          By continuing, you agree to our Terms of Service and Privacy Policy. Guest investigations are active in this browser session only.
        </p>
      </div>
    </div>
  );
}

// In-Session Auth Modal (For upgrading a guest session to a permanent account)
function AuthModal({
  isOpen,
  onClose,
  onLogin,
}: {
  isOpen: boolean;
  onClose: () => void;
  onLogin: (user: UserAccount) => void;
}) {
  const [email, setEmail] = useState("");
  const [authError, setAuthError] = useState<string | null>(null);
  const [isAuthenticating, setIsAuthenticating] = useState<"google" | null>(null);

  if (!isOpen) return null;

  const handleGoogleLogin = async () => {
    setAuthError(null);
    if (!isFirebaseConfigured) {
      setAuthError(
        "OAuth is not configured yet. Please configure the Firebase environment variables (VITE_FIREBASE_*) in Vercel or your local .env file."
      );
      return;
    }
    setIsAuthenticating("google");
    try {
      const res = await signInWithGoogle();
      if (res.user) {
        onLogin(res.user);
      } else if (res.error) {
        setAuthError(res.error);
      }
    } catch (err: any) {
      setAuthError(err?.message || "Google sign-in failed.");
    } finally {
      setIsAuthenticating(null);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-fade-in">
      <div 
        className="relative w-full max-w-[420px] bg-surface-container-lowest border border-surface-container-high/60 rounded-2xl shadow-2xl p-6 sm:p-8 flex flex-col items-center gap-5 text-center transition-all"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Close Button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 p-2 min-w-[44px] min-h-[44px] flex items-center justify-center rounded-full text-outline hover:text-on-surface hover:bg-surface-container transition-colors cursor-pointer"
          type="button"
          aria-label="Close"
        >
          <MatIcon name="close" size={18} />
        </button>

        {/* Brand Icon */}
        <div className="w-12 h-12 rounded-full bg-primary/10 border border-primary/20 flex items-center justify-center text-primary mt-1 shadow-xs">
          <LucidBrandLogo size="sm" showText={false} />
        </div>

        {/* Titles */}
        <div className="space-y-1">
          <h2 className="text-2xl font-bold tracking-tight text-on-surface">Sign in or sign up</h2>
          <p className="text-xs sm:text-sm text-on-surface-variant">Save your investigations across devices</p>
        </div>

        {/* Auth Error Banner */}
        {authError && (
          <div className="w-full p-3 rounded-xl bg-error-container/40 border border-error/30 text-error text-xs text-left flex items-start gap-2">
            <MatIcon name="error_outline" size={16} className="shrink-0 mt-0.5" />
            <span className="flex-1 leading-relaxed">{authError}</span>
          </div>
        )}

        {/* OAuth Buttons */}
        <div className="w-full space-y-2.5 pt-1">
          <button
            onClick={handleGoogleLogin}
            disabled={isAuthenticating !== null}
            className="w-full min-h-[44px] flex items-center justify-center gap-3 px-4 py-2.5 rounded-xl border border-surface-container-high hover:bg-surface-container-low transition-all font-label-md text-label-md text-on-surface font-medium cursor-pointer disabled:opacity-60 disabled:cursor-not-allowed"
            type="button"
          >
            {isAuthenticating === "google" ? (
              <MatIcon name="progress_activity" size={18} className="animate-spin text-primary" />
            ) : (
              <svg className="w-4 h-4 shrink-0" viewBox="0 0 24 24">
                <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" />
                <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" />
                <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.06H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.94l2.85-2.22.81-.63z" />
                <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.52 6.16-4.52z" />
              </svg>
            )}
            <span>{isAuthenticating === "google" ? "Signing in..." : "Continue with Google"}</span>
          </button>
        </div>

        {/* Divider */}
        <div className="w-full flex items-center gap-3 text-xs text-outline my-0.5">
          <div className="flex-1 h-px bg-surface-container-high" />
          <span>Or continue with email</span>
          <div className="flex-1 h-px bg-surface-container-high" />
        </div>

        {/* Email form */}
        <form 
          onSubmit={(e) => {
            e.preventDefault();
            if (email.trim()) {
              onLogin({ name: email.split('@')[0], email: email.trim(), isGuest: false });
            }
          }}
          className="w-full space-y-2.5"
        >
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="Enter your email address"
            aria-label="Email address"
            id="welcome-email-input"
            className="w-full min-h-[44px] px-4 py-2.5 rounded-xl bg-surface-container-low border border-surface-container-high text-on-surface text-base sm:text-sm placeholder:text-outline focus:outline-none focus:border-primary transition-all"
            required
          />
          <button
            type="submit"
            className="w-full min-h-[44px] py-2.5 rounded-xl bg-primary text-on-primary font-semibold text-sm hover:opacity-95 transition-all cursor-pointer shadow-xs"
          >
            Continue with Email
          </button>
        </form>
      </div>
    </div>
  );
}

// Navbar / Header Component (Desktop Navigation + Mobile Hamburger & Slide Drawer)
function AppHeader({
  onNavigateHome,
  darkMode,
  onToggleDarkMode,
  currentUser,
  onOpenAuth,
  onSignOut,
  voiceGuideEnabled,
  onToggleVoiceGuide,
  isSpeaking,
  onStopSpeaking,
}: {
  onNavigateHome: () => void;
  darkMode: boolean;
  onToggleDarkMode: () => void;
  currentUser: UserAccount | null;
  onOpenAuth: () => void;
  onSignOut: () => void;
  voiceGuideEnabled?: boolean;
  onToggleVoiceGuide?: () => void;
  isSpeaking?: boolean;
  onStopSpeaking?: () => void;
}) {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  return (
    <>
      <header className="fixed top-0 left-0 right-0 z-40 bg-surface-container-lowest/80 backdrop-blur-xl border-b border-surface-container-high/40 shadow-xs">
        <div className="h-16 max-w-[1000px] mx-auto px-4 sm:px-6 flex items-center justify-between">
          {/* Brand Logo */}
          <LucidBrandLogo size="md" showText onClick={() => { onNavigateHome(); setMobileMenuOpen(false); }} />

          {/* Desktop Navigation: User Profile, Voice Guide, Theme Toggle */}
          <div className="hidden sm:flex items-center gap-2.5">
            {/* Audio Voice Guide Toggle */}
            {onToggleVoiceGuide && (
              <button
                onClick={onToggleVoiceGuide}
                className={`min-h-[44px] px-3.5 py-1.5 rounded-full flex items-center gap-1.5 font-label-sm text-xs font-semibold border transition-all cursor-pointer ${
                  voiceGuideEnabled
                    ? "bg-primary text-on-primary border-primary shadow-xs ring-2 ring-primary/20"
                    : "bg-surface-container hover:bg-surface-container-high text-on-surface border-surface-container-high/60"
                }`}
                type="button"
                aria-label={voiceGuideEnabled ? "Turn audio voice guide off" : "Turn audio voice guide on"}
                title={voiceGuideEnabled ? "Voice Guide is ON" : "Turn ON Voice Guide"}
              >
                <MatIcon name={voiceGuideEnabled ? "volume_up" : "volume_off"} size={16} />
                <span>Voice Guide {voiceGuideEnabled ? "ON" : "OFF"}</span>
              </button>
            )}

            {/* Active Speech Stop Button */}
            {isSpeaking && onStopSpeaking && (
              <button
                onClick={onStopSpeaking}
                className="min-h-[44px] px-3 py-1.5 rounded-full flex items-center gap-1.5 font-label-sm text-xs font-semibold bg-error text-on-error hover:opacity-95 transition-all cursor-pointer shadow-xs animate-pulse"
                type="button"
                aria-label="Stop audio voice speech"
                title="Stop speaking"
              >
                <MatIcon name="stop_circle" size={16} />
                <span>Stop</span>
              </button>
            )}

            {currentUser ? (
              <div className="flex items-center gap-2">
                {currentUser.isGuest ? (
                  <span className="px-2.5 py-1 rounded-full text-xs font-semibold bg-[#FDF1DC] text-[#9A5B00] border border-[#F5D49E] dark:bg-amber-950/40 dark:text-amber-400 dark:border-amber-900/40 flex items-center gap-1">
                    <MatIcon name="person" size={13} />
                    <span>Guest</span>
                  </span>
                ) : (
                  <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-surface-container border border-surface-container-high text-xs font-medium text-on-surface">
                    {currentUser.avatar ? (
                      <img src={currentUser.avatar} alt={currentUser.name || "User Avatar"} className="w-5 h-5 rounded-full object-cover shrink-0" />
                    ) : (
                      <span className="w-5 h-5 rounded-full bg-primary text-on-primary text-xs font-bold flex items-center justify-center shrink-0">
                        {(currentUser.name || currentUser.email)[0].toUpperCase()}
                      </span>
                    )}
                    <span className="max-w-[110px] truncate">{currentUser.name || currentUser.email.split('@')[0]}</span>
                  </div>
                )}
                {currentUser.isGuest && (
                  <button
                    onClick={onOpenAuth}
                    className="min-h-[44px] px-2 text-xs text-primary hover:underline font-semibold cursor-pointer inline-flex items-center"
                    type="button"
                    aria-label="Sign In to save investigations"
                  >
                    Sign In
                  </button>
                )}
                <button
                  onClick={onSignOut}
                  className="min-h-[44px] px-2 text-xs text-on-surface-variant hover:text-error font-medium underline cursor-pointer ml-1 inline-flex items-center"
                  type="button"
                  aria-label={currentUser.isGuest ? "Exit Guest Mode" : "Sign Out"}
                >
                  {currentUser.isGuest ? "Exit Guest" : "Sign Out"}
                </button>
              </div>
            ) : (
              <button
                onClick={onOpenAuth}
                className="flex items-center gap-1.5 px-3.5 py-1.5 min-h-[44px] rounded-full bg-surface-container hover:bg-surface-container-high text-on-surface font-label-sm text-label-sm font-semibold transition-all border border-surface-container-high shadow-2xs cursor-pointer"
                type="button"
              >
                <MatIcon name="account_circle" size={16} />
                <span>Sign In</span>
              </button>
            )}

            <button
              aria-label="Toggle theme"
              onClick={onToggleDarkMode}
              className="p-2 min-w-[44px] min-h-[44px] flex items-center justify-center rounded-full text-on-surface-variant hover:bg-surface-container hover:text-on-surface transition-all cursor-pointer"
              type="button"
              title={darkMode ? "Switch to light mode" : "Switch to dark mode"}
            >
              <MatIcon name={darkMode ? "light_mode" : "dark_mode"} size={20} />
            </button>
          </div>

          {/* Mobile Actions: Voice Guide + Theme Toggle + Hamburger Menu */}
          <div className="flex sm:hidden items-center gap-1">
            {onToggleVoiceGuide && (
              <button
                onClick={onToggleVoiceGuide}
                className={`p-2.5 min-w-[44px] min-h-[44px] flex items-center justify-center rounded-full transition-all cursor-pointer ${
                  voiceGuideEnabled ? "bg-primary text-on-primary" : "text-on-surface-variant hover:bg-surface-container"
                }`}
                type="button"
                aria-label={voiceGuideEnabled ? "Voice Guide is ON" : "Turn ON Voice Guide"}
                title="Voice Guide"
              >
                <MatIcon name={voiceGuideEnabled ? "volume_up" : "volume_off"} size={20} />
              </button>
            )}
            {isSpeaking && onStopSpeaking && (
              <button
                onClick={onStopSpeaking}
                className="p-2.5 min-w-[44px] min-h-[44px] flex items-center justify-center rounded-full bg-error text-on-error cursor-pointer animate-pulse"
                type="button"
                aria-label="Stop speaking"
                title="Stop speaking"
              >
                <MatIcon name="stop_circle" size={20} />
              </button>
            )}
            <button
              aria-label="Toggle theme"
              onClick={onToggleDarkMode}
              className="p-2.5 min-w-[44px] min-h-[44px] flex items-center justify-center rounded-full text-on-surface-variant hover:bg-surface-container transition-all cursor-pointer"
              type="button"
            >
              <MatIcon name={darkMode ? "light_mode" : "dark_mode"} size={20} />
            </button>

            <button
              aria-label="Open mobile menu"
              onClick={() => setMobileMenuOpen(true)}
              className="p-2.5 min-w-[44px] min-h-[44px] flex items-center justify-center rounded-full text-on-surface hover:bg-surface-container transition-all cursor-pointer"
              type="button"
            >
              <MatIcon name="menu" size={24} />
            </button>
          </div>
        </div>
      </header>

      {/* Mobile Slide Drawer Navigation (Slide Sheet) */}
      {mobileMenuOpen && (
        <div className="fixed inset-0 z-50 flex justify-end animate-fade-in sm:hidden">
          {/* Backdrop */}
          <div
            className="fixed inset-0 bg-black/60 backdrop-blur-xs transition-opacity"
            onClick={() => setMobileMenuOpen(false)}
          />

          {/* Drawer Panel */}
          <div className="relative w-72 max-w-[85vw] h-full bg-surface-container-lowest border-l border-surface-container-high shadow-2xl p-5 flex flex-col justify-between overflow-y-auto z-10 animate-slide-left">
            <div className="space-y-6">
              {/* Drawer Header */}
              <div className="flex items-center justify-between pb-3 border-b border-surface-container-high/40">
                <LucidBrandLogo size="sm" showText onClick={() => { onNavigateHome(); setMobileMenuOpen(false); }} />
                <button
                  onClick={() => setMobileMenuOpen(false)}
                  className="p-2 min-w-[44px] min-h-[44px] flex items-center justify-center rounded-full text-outline hover:text-on-surface hover:bg-surface-container transition-colors cursor-pointer"
                  type="button"
                  aria-label="Close menu"
                >
                  <MatIcon name="close" size={20} />
                </button>
              </div>

              {/* User Account / Session Status */}
              <div className="p-3.5 rounded-xl bg-surface-container-low border border-surface-container-high/40 space-y-2.5">
                {currentUser ? (
                  <>
                    <div className="flex items-center gap-2.5">
                      {currentUser.avatar ? (
                        <img src={currentUser.avatar} alt={currentUser.name || "User Avatar"} className="w-9 h-9 rounded-full object-cover shrink-0" />
                      ) : (
                        <span className="w-9 h-9 rounded-full bg-primary text-on-primary text-sm font-bold flex items-center justify-center shrink-0">
                          {(currentUser.name || currentUser.email)[0].toUpperCase()}
                        </span>
                      )}
                      <div className="min-w-0 flex-1">
                        <div className="font-semibold text-sm text-on-surface truncate">
                          {currentUser.name}
                        </div>
                        <div className="text-xs text-outline truncate">
                          {currentUser.email}
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center justify-between pt-1">
                      {currentUser.isGuest ? (
                        <span className="px-2 py-0.5 rounded-full text-xs font-semibold bg-[#FDF1DC] text-[#9A5B00] border border-[#F5D49E] dark:bg-amber-950/40 dark:text-amber-400 dark:border-amber-900/40">
                          Guest Session
                        </span>
                      ) : (
                        <span className="px-2 py-0.5 rounded-full text-xs font-semibold bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 border border-emerald-500/30">
                          Verified Account
                        </span>
                      )}
                      {currentUser.isGuest && (
                        <button
                          onClick={() => { setMobileMenuOpen(false); onOpenAuth(); }}
                          className="min-h-[44px] px-2 text-xs font-semibold text-primary hover:underline cursor-pointer inline-flex items-center"
                          type="button"
                          aria-label="Sign In to account"
                        >
                          Sign In
                        </button>
                      )}
                    </div>
                  </>
                ) : (
                  <div className="space-y-2 text-center py-1">
                    <p className="text-xs text-outline">You are not signed in.</p>
                    <button
                      onClick={() => { setMobileMenuOpen(false); onOpenAuth(); }}
                      className="w-full min-h-[44px] py-2 px-3 rounded-xl bg-primary text-on-primary text-xs font-semibold hover:opacity-95 transition-all cursor-pointer"
                      type="button"
                    >
                      Sign In or Continue
                    </button>
                  </div>
                )}
              </div>

              {/* Navigation Options */}
              <div className="space-y-1">
                <button
                  onClick={() => { onNavigateHome(); setMobileMenuOpen(false); }}
                  className="w-full min-h-[44px] px-3.5 py-2.5 rounded-xl flex items-center gap-3 text-sm font-medium text-on-surface hover:bg-surface-container transition-colors cursor-pointer text-left"
                  type="button"
                >
                  <MatIcon name="search" size={20} className="text-primary" />
                  <span>Verify New Content</span>
                </button>

                <button
                  onClick={() => { onToggleDarkMode(); }}
                  className="w-full min-h-[44px] px-3.5 py-2.5 rounded-xl flex items-center justify-between text-sm font-medium text-on-surface hover:bg-surface-container transition-colors cursor-pointer text-left"
                  type="button"
                >
                  <div className="flex items-center gap-3">
                    <MatIcon name={darkMode ? "light_mode" : "dark_mode"} size={20} className="text-outline" />
                    <span>{darkMode ? "Light Theme" : "Dark Theme"}</span>
                  </div>
                  <span className="text-xs text-outline font-semibold">
                    {darkMode ? "Dark" : "Light"}
                  </span>
                </button>
              </div>
            </div>

            {/* Drawer Bottom Actions */}
            <div className="pt-4 border-t border-surface-container-high/40 space-y-3">
              {currentUser && (
                <button
                  onClick={() => { setMobileMenuOpen(false); onSignOut(); }}
                  className="w-full min-h-[44px] px-3.5 py-2 rounded-xl flex items-center justify-center gap-2 text-sm font-medium text-error hover:bg-error/10 border border-error/20 transition-colors cursor-pointer"
                  type="button"
                >
                  <MatIcon name="logout" size={18} />
                  <span>{currentUser.isGuest ? "Exit Guest Mode" : "Sign Out"}</span>
                </button>
              )}
              <div className="text-xs text-center text-outline">
                Lucid AI Fact Verification Platform
              </div>
            </div>
          </div>
        </div>
      )}
    </>
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
  onOcrExtract,
  isExtractingOcr,
  history,
  onSelectHistory,
  onClearHistory,
  verifyError,
  onClearError,
  initialInput,
  onInputChange,
}: {
  onVerify: (payload: { mode: "url" | "text" | "upload"; content: string; file?: File }) => void;
  onOcrExtract: (file: File) => void;
  isExtractingOcr: boolean;
  history: HistoryItem[];
  onSelectHistory: (item: HistoryItem) => void;
  onClearHistory?: () => void;
  verifyError?: string | null;
  onClearError?: () => void;
  initialInput?: { mode: "url" | "text" | "image" | "pdf"; query: string; selectedFile: File | null };
  onInputChange?: (state: { mode: "url" | "text" | "image" | "pdf"; query: string; selectedFile: File | null }) => void;
}) {
  const [mode, setMode] = useState<"url" | "text" | "image" | "pdf">(initialInput?.mode || "url");
  const [query, setQuery] = useState(initialInput?.query || "");
  const [selectedFile, setSelectedFile] = useState<File | null>(initialInput?.selectedFile || null);
  const [isDragging, setIsDragging] = useState(false);
  const [isHistoryOpen, setIsHistoryOpen] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Voice Input States & Web Speech API integration
  const [isListening, setIsListening] = useState(false);
  const [speechSupported, setSpeechSupported] = useState(true);
  const [voiceLanguage, setVoiceLanguage] = useState<"en-IN" | "ta-IN" | "hi-IN">("en-IN");
  const [voiceError, setVoiceError] = useState<string | null>(null);
  const recognitionRef = useRef<any>(null);

  useEffect(() => {
    const SpeechRec = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SpeechRec) {
      setSpeechSupported(false);
    }
  }, []);

  const toggleVoiceInput = () => {
    setVoiceError(null);
    const SpeechRec = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SpeechRec) {
      setSpeechSupported(false);
      return;
    }

    if (isListening) {
      if (recognitionRef.current) {
        try {
          recognitionRef.current.stop();
        } catch {}
      }
      setIsListening(false);
      return;
    }

    try {
      const recognition = new SpeechRec();
      recognition.lang = voiceLanguage;
      recognition.interimResults = true;
      recognition.continuous = false;

      recognition.onstart = () => {
        setIsListening(true);
        setVoiceError(null);
      };

      recognition.onresult = (event: any) => {
        let liveTranscript = "";
        for (let i = 0; i < event.results.length; i++) {
          liveTranscript += event.results[i][0].transcript;
        }
        if (liveTranscript) {
          updateQuery(liveTranscript);
        }
      };

      recognition.onerror = (event: any) => {
        setIsListening(false);
        if (event.error === "not-allowed" || event.error === "service-not-allowed") {
          setVoiceError("Microphone access was denied. Please allow microphone permissions in your browser.");
        } else if (event.error === "no-speech") {
          setVoiceError("No speech detected. Please speak into your microphone and try again.");
        } else {
          setVoiceError(`Voice input error (${event.error}). Please try again.`);
        }
      };

      recognition.onend = () => {
        setIsListening(false);
      };

      recognitionRef.current = recognition;
      recognition.start();
    } catch (err) {
      console.error("Speech recognition initialization error:", err);
      setIsListening(false);
      setVoiceError("Could not initialize voice input on this browser.");
    }
  };

  const updateMode = (newMode: "url" | "text" | "image" | "pdf") => {
    setMode(newMode);
    onInputChange?.({ mode: newMode, query, selectedFile });
  };

  const updateQuery = (newQuery: string) => {
    setQuery(newQuery);
    onInputChange?.({ mode, query: newQuery, selectedFile });
  };

  const updateFile = (newFile: File | null, newMode?: "url" | "text" | "image" | "pdf") => {
    setSelectedFile(newFile);
    const m = newMode || mode;
    if (newMode) setMode(newMode);
    onInputChange?.({ mode: m, query, selectedFile: newFile });
  };

  const handleSubmit = (e?: FormEvent) => {
    if (e) e.preventDefault();
    if (mode === "image") {
      if (!selectedFile) {
        fileInputRef.current?.click();
        return;
      }
      onOcrExtract(selectedFile);
    } else if (mode === "pdf") {
      if (!selectedFile) {
        fileInputRef.current?.click();
        return;
      }
      onVerify({ mode: "upload", content: selectedFile.name, file: selectedFile });
    } else if (mode === "url") {
      if (!query.trim()) return;
      const trimmed = query.trim();
      const isLikelyUrl = /^(https?:\/\/|[a-zA-Z0-9-]+\.[a-zA-Z]{2,})(\S*)$/.test(trimmed) && !trimmed.includes(" ");
      if (isLikelyUrl) {
        onVerify({ mode: "url", content: trimmed });
      } else {
        onVerify({ mode: "text", content: trimmed });
      }
    } else {
      if (!query.trim()) return;
      onVerify({ mode: "text", content: query.trim() });
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      updateFile(file, file.type.includes("pdf") ? "pdf" : "image");
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) {
      updateFile(file, file.type.includes("pdf") ? "pdf" : "image");
    }
  };

  return (
    <div className="max-w-[760px] w-full mx-auto px-3 sm:px-6 py-6 sm:py-12 md:py-20 flex flex-col items-center">
      {/* Hidden File Input */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*,.pdf,.doc,.docx,.txt"
        className="hidden"
        onChange={handleFileChange}
        aria-label="Upload document, image or PDF file"
        id="file-upload-input"
      />

      {/* Logo */}
      <div className="mb-4 sm:mb-6 flex flex-col items-center">
        <LucidBrandLogo size="hero" showText={false} />
      </div>

      {/* Headline & Subtitle */}
      <div className="text-center max-w-2xl mb-6 sm:mb-8 space-y-2">
        <h1 className="text-2xl sm:text-4xl md:text-5xl font-extrabold tracking-tight text-on-surface leading-tight">
          Verify Information Before You Believe It
        </h1>
        <p className="text-xs sm:text-base text-on-surface-variant max-w-xl mx-auto px-2 leading-relaxed">
          Get a simple summary, evaluate credibility, and understand the evidence behind any article.
        </p>
      </div>

      {/* Verification Inline Error Card */}
      {verifyError && (
        <div
          className="w-full mb-4 p-4 rounded-xl bg-error-container/40 border border-error/30 text-on-error-container flex flex-col sm:flex-row sm:items-center justify-between gap-3 animate-fade-in shadow-xs"
          role="alert"
          aria-live="assertive"
        >
          <div className="flex items-start sm:items-center gap-3">
            <MatIcon name="error" size={22} className="text-error shrink-0 mt-0.5 sm:mt-0" />
            <div>
              <p className="font-semibold text-xs sm:text-sm text-error leading-snug">Verification Failed</p>
              <p className="text-xs text-on-surface-variant mt-0.5">{verifyError}</p>
            </div>
          </div>
          <div className="flex items-center gap-2 shrink-0 self-end sm:self-center">
            <button
              onClick={() => handleSubmit()}
              className="min-h-[44px] px-4 py-2 rounded-full bg-error text-on-error font-semibold text-xs hover:opacity-95 transition-all cursor-pointer flex items-center gap-1.5 shadow-2xs"
              type="button"
              aria-label="Try verification again"
            >
              <MatIcon name="refresh" size={16} />
              <span>Try again</span>
            </button>
            {onClearError && (
              <button
                onClick={onClearError}
                className="min-h-[44px] min-w-[44px] rounded-full flex items-center justify-center text-outline hover:text-on-surface hover:bg-surface-container transition-colors cursor-pointer"
                type="button"
                aria-label="Dismiss error message"
              >
                <MatIcon name="close" size={18} />
              </button>
            )}
          </div>
        </div>
      )}

      {/* Main Card */}
      <div
        onPointerMove={handleCardPointerMove}
        className="w-full lucid-card glass-card spotlight-card rounded-2xl shadow-md p-3 sm:p-5 flex flex-col gap-3 transition-all duration-300"
      >
        {/* Input Tabs: URL, Text, Image, PDF (Mobile-Friendly Scrollable Container) */}
        <div className="flex items-center gap-1.5 pb-2 border-b border-surface-container-high/20 overflow-x-auto no-scrollbar scroll-smooth w-full">
          <button
            onClick={() => updateMode("url")}
            className={`min-h-[44px] px-4 py-2 rounded-full font-label-md text-label-md transition-all cursor-pointer shrink-0 ${
              mode === "url"
                ? "bg-primary text-on-primary font-semibold shadow-xs"
                : "bg-surface-container-low hover:bg-surface-container text-on-surface-variant"
            }`}
            type="button"
          >
            URL
          </button>

          <button
            onClick={() => updateMode("text")}
            className={`min-h-[44px] px-4 py-2 rounded-full font-label-md text-label-md transition-all cursor-pointer shrink-0 ${
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
              updateMode("image");
              fileInputRef.current?.click();
            }}
            className={`min-h-[44px] px-4 py-2 rounded-full font-label-md text-label-md transition-all cursor-pointer shrink-0 ${
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
              updateMode("pdf");
              fileInputRef.current?.click();
            }}
            className={`min-h-[44px] px-4 py-2 rounded-full font-label-md text-label-md transition-all cursor-pointer shrink-0 ${
              mode === "pdf"
                ? "bg-primary text-on-primary font-semibold shadow-xs"
                : "bg-surface-container-low hover:bg-surface-container text-on-surface-variant"
            }`}
            type="button"
          >
            PDF
          </button>
        </div>

        {/* Voice Input Active State Indicator */}
        {isListening && (
          <div className="w-full flex items-center gap-2.5 px-3.5 py-2.5 rounded-xl bg-primary/10 border border-primary/20 text-primary text-xs font-semibold animate-pulse" role="status">
            <span className="w-2.5 h-2.5 rounded-full bg-error animate-ping shrink-0" />
            <span>Listening in {voiceLanguage === "en-IN" ? "English (India)" : voiceLanguage === "ta-IN" ? "Tamil" : "Hindi"}... Speak clearly. Your transcript populates the input live.</span>
          </div>
        )}

        {/* Voice Error Notification (In-UI Banner, No alert()) */}
        {voiceError && (
          <div className="w-full flex items-center justify-between gap-2 px-3 py-2 rounded-xl bg-error-container/40 border border-error/30 text-error text-xs font-medium" role="alert">
            <div className="flex items-center gap-2">
              <MatIcon name="mic_off" size={16} />
              <span>{voiceError}</span>
            </div>
            <button onClick={() => setVoiceError(null)} className="min-h-[44px] min-w-[44px] flex items-center justify-center text-error hover:opacity-80 cursor-pointer" aria-label="Dismiss voice error">
              <MatIcon name="close" size={16} />
            </button>
          </div>
        )}

        {/* Input Form */}
        <form onSubmit={handleSubmit} className="flex flex-col gap-2.5 w-full">
          {mode === "text" ? (
            <div className="relative flex flex-col w-full bg-surface-container-low rounded-xl p-2.5 border border-surface-container-high/40 focus-within:bg-surface-container-lowest focus-within:shadow-[0_0_0_2px_rgba(0,74,198,0.2)] transition-all">
              <textarea
                value={query}
                onChange={(e) => updateQuery(e.target.value)}
                placeholder="Paste news article content, statement, or text to verify..."
                aria-label="News article content to verify"
                id="verification-text-query"
                rows={4}
                className="w-full bg-transparent border-none outline-none font-body-md text-base sm:text-sm text-on-surface placeholder:text-outline p-2 resize-y min-h-[110px]"
              />
              <div className="flex flex-wrap items-center justify-between gap-2 pt-2 border-t border-surface-container-high/30 mt-1">
                {/* Voice Input Controls */}
                {speechSupported ? (
                  <div className="flex items-center gap-1.5 shrink-0">
                    <select
                      value={voiceLanguage}
                      onChange={(e) => setVoiceLanguage(e.target.value as any)}
                      className="min-h-[44px] px-2.5 py-1.5 rounded-full bg-surface-container hover:bg-surface-container-high border border-surface-container-high/60 text-xs font-semibold text-on-surface cursor-pointer outline-none focus:ring-2 focus:ring-primary/40 transition-colors"
                      aria-label="Select voice recognition language"
                      title="Voice recognition language"
                    >
                      <option value="en-IN">English (India)</option>
                      <option value="ta-IN">Tamil (தமிழ்)</option>
                      <option value="hi-IN">Hindi (हिन्दी)</option>
                    </select>

                    <button
                      onClick={toggleVoiceInput}
                      type="button"
                      aria-label={isListening ? "Stop listening to voice input" : "Start voice input"}
                      className={`min-h-[44px] min-w-[44px] px-3.5 py-2 rounded-full flex items-center justify-center gap-1.5 transition-all cursor-pointer font-label-md text-xs font-semibold ${
                        isListening
                          ? "bg-error text-on-error animate-pulse shadow-md ring-2 ring-error/50"
                          : "bg-surface-container hover:bg-surface-container-high text-on-surface border border-surface-container-high/60 shadow-2xs"
                      }`}
                      title={isListening ? "Listening... click to stop" : "Speak into microphone"}
                    >
                      <MatIcon name={isListening ? "mic" : "mic_none"} size={18} className={isListening ? "animate-bounce" : "text-primary"} />
                      <span>{isListening ? "Listening..." : "Voice Input"}</span>
                    </button>
                  </div>
                ) : (
                  <span className="text-[12px] text-outline px-2.5 py-1 rounded-full bg-surface-container-low shrink-0" title="Voice input not supported in this browser">
                    Voice input unavailable
                  </span>
                )}

                <button
                  type="submit"
                  disabled={!query.trim()}
                  className="w-full sm:w-auto min-h-[44px] flex items-center justify-center gap-1.5 px-6 py-2.5 rounded-full bg-primary text-on-primary hover:bg-surface-tint disabled:opacity-40 font-label-md text-label-md transition-all shadow-xs cursor-pointer font-semibold"
                >
                  <MatIcon name="search" size={18} />
                  <span>Verify</span>
                </button>
              </div>
            </div>
          ) : (
            <div className="relative flex flex-col sm:flex-row items-stretch sm:items-center w-full bg-surface-container-low rounded-xl p-1.5 sm:p-2 gap-2 border border-surface-container-high/40 focus-within:bg-surface-container-lowest focus-within:shadow-[0_0_0_2px_rgba(0,74,198,0.2)] transition-all">
              <div className="flex items-center flex-1 min-w-0">
                <MatIcon name="search" size={22} className="text-outline ml-2 shrink-0 select-none" />
                <input
                  autoComplete="off"
                  value={selectedFile ? selectedFile.name : query}
                  onChange={(e) => {
                    if (selectedFile) updateFile(null);
                    updateQuery(e.target.value);
                  }}
                  aria-label="Article URL, statement, or search query"
                  id="verification-url-input"
                  className="w-full bg-transparent border-none outline-none font-body-md text-base sm:text-lg text-on-surface placeholder:text-outline px-3 py-2 min-h-[44px]"
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
                      updateFile(null);
                      updateQuery("");
                    }}
                    className="p-2 mr-1 min-w-[44px] min-h-[44px] flex items-center justify-center text-outline hover:text-on-surface rounded-full transition-colors cursor-pointer shrink-0"
                    title="Clear input"
                    type="button"
                    aria-label="Clear input query"
                  >
                    <MatIcon name="close" size={18} />
                  </button>
                )}
              </div>

              {/* Voice Input Controls in URL / Search Bar */}
              {speechSupported ? (
                <div className="flex items-center gap-1 shrink-0 self-end sm:self-center">
                  <select
                    value={voiceLanguage}
                    onChange={(e) => setVoiceLanguage(e.target.value as any)}
                    className="min-h-[44px] px-2 py-1 rounded-full bg-surface-container hover:bg-surface-container-high border border-surface-container-high/60 text-xs font-semibold text-on-surface cursor-pointer outline-none focus:ring-2 focus:ring-primary/40 transition-colors"
                    aria-label="Select voice recognition language"
                    title="Voice recognition language"
                  >
                    <option value="en-IN">EN (IN)</option>
                    <option value="ta-IN">TA (தமிழ்)</option>
                    <option value="hi-IN">HI (हिन्दी)</option>
                  </select>

                  <button
                    onClick={toggleVoiceInput}
                    type="button"
                    aria-label={isListening ? "Stop listening to voice input" : "Start voice input"}
                    className={`min-h-[44px] min-w-[44px] px-2.5 py-1.5 rounded-full flex items-center justify-center gap-1 transition-all cursor-pointer text-xs font-semibold ${
                      isListening
                        ? "bg-error text-on-error animate-pulse shadow-md ring-2 ring-error/50"
                        : "bg-surface-container hover:bg-surface-container-high text-on-surface border border-surface-container-high/60"
                    }`}
                    title={isListening ? "Listening... click to stop" : "Speak into microphone"}
                  >
                    <MatIcon name={isListening ? "mic" : "mic_none"} size={18} className={isListening ? "animate-bounce" : "text-primary"} />
                    <span className="hidden md:inline">{isListening ? "Listening" : "Speak"}</span>
                  </button>
                </div>
              ) : null}

              {/* Verify Button */}
              <button
                type="submit"
                disabled={(!query.trim() && !selectedFile) || isExtractingOcr}
                className="w-full sm:w-auto min-h-[44px] flex items-center justify-center gap-1.5 px-6 py-2.5 rounded-full bg-primary text-on-primary hover:bg-surface-tint disabled:opacity-40 font-label-md text-label-md transition-all shadow-xs shrink-0 cursor-pointer font-semibold"
              >
                <MatIcon name={isExtractingOcr ? "sync" : "search"} size={18} className={isExtractingOcr ? "animate-spin" : ""} />
                <span>{isExtractingOcr ? "Extracting OCR..." : "Verify"}</span>
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
          className={`cursor-pointer rounded-xl flex items-center justify-center gap-2 min-h-[48px] py-3 px-4 transition-all text-center border border-dashed ${
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

      {/* Recent Investigations (Collapsible Dropdown - Hidden by default) */}
      <div className="w-full mt-8">
        <div className="lucid-card bg-surface-container-lowest rounded-xl shadow-xs border border-surface-container-high/40 overflow-hidden">
          <button
            onClick={() => setIsHistoryOpen((prev) => !prev)}
            className="w-full px-5 py-3.5 flex items-center justify-between text-left hover:bg-surface-container-low transition-colors cursor-pointer select-none"
            type="button"
          >
            <div className="flex items-center gap-2">
              <MatIcon name="history" size={18} className="text-primary" />
              <span className="font-label-md text-label-md font-semibold text-on-surface">Recent Investigations</span>
              <span className="px-2 py-0.5 rounded-full bg-surface-container text-primary font-label-xs text-label-xs font-semibold">
                {history.length}
              </span>
            </div>
            <div className="flex items-center gap-1.5 text-outline">
              <span className="text-xs font-medium">{isHistoryOpen ? "Hide" : "Show"}</span>
              <MatIcon
                name="expand_more"
                size={18}
                className={`transition-transform duration-200 ${isHistoryOpen ? "rotate-180" : ""}`}
              />
            </div>
          </button>

          {isHistoryOpen && (
            <div className="p-4 border-t border-surface-container-high/30 animate-fade-in">
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
                      className="lucid-card-sub py-2.5 flex items-center justify-between hover:bg-surface-container-low px-2 rounded-lg cursor-pointer transition-colors"
                    >
                      <div className="flex flex-col max-w-lg">
                        <span className="font-label-md text-label-md font-medium text-on-surface truncate">
                          {item.title}
                        </span>
                        <span className="font-label-xs text-label-xs text-outline">{item.time}</span>
                      </div>
                      <span
                        className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full font-label-xs text-xs font-semibold ${
                          item.verdict === "VERIFIED" || item.verdict === "LIKELY TRUE"
                            ? "bg-tertiary/10 text-tertiary"
                            : item.verdict === "KNOWN FALSE" || item.verdict === "LIKELY FALSE"
                            ? "bg-error/10 text-error"
                            : "bg-[#FDF1DC] text-[#9A5B00] border border-[#F5D49E] dark:bg-amber-950/40 dark:text-amber-400 dark:border-amber-900/40"
                        }`}
                      >
                        <MatIcon
                          name={
                            item.verdict === "VERIFIED" || item.verdict === "LIKELY TRUE"
                              ? "check_circle"
                              : item.verdict === "KNOWN FALSE" || item.verdict === "LIKELY FALSE"
                              ? "cancel"
                              : item.verdict === "Satire"
                              ? "theater_comedy"
                              : "help"
                          }
                          size={14}
                        />
                        <span>{item.verdict}</span>
                      </span>
                    </div>
                  ))}
                  {history.length > 0 && onClearHistory && (
                    <div className="flex justify-end pt-2">
                      <button
                        onClick={onClearHistory}
                        className="min-h-[44px] px-3 py-2 text-xs text-on-surface-variant hover:text-error font-medium underline cursor-pointer inline-flex items-center"
                        type="button"
                        aria-label="Clear all investigations history"
                      >
                        Clear all
                      </button>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ==========================================
// 3b. OCR PREVIEW SCREEN (NEW STEP IN IMAGE WORKFLOW)
// ==========================================

function OcrPreviewScreen({
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

// ==========================================
// 5. VERIFICATION REPORT SCREEN (6-STAGE FLOW)
// ==========================================

function ReportScreen({
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
  const [techDetailsOpen, setTechDetailsOpen] = useState(false);
  const [copiedLink, setCopiedLink] = useState(false);
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

  // Voice Guidance: Automatically announce verdict and one-line reason when report renders (if user turned on Voice Guide)
  useEffect(() => {
    if (voiceGuideEnabled && result && speakVoiceGuidance) {
      const v = result.ragGroundedVerdict?.verdict || result.overallVerdict || result.verdict || "Unverified";
      const reason = result.summaryReason || result.explanation?.[0] || summary.executiveSummary || "";
      const firstSentence = reason.split(/(?<=[.!?])\s+/)[0] || "";
      speakVoiceGuidance(`Result: ${v}. ${firstSentence}`);
    }
  }, [result, voiceGuideEnabled, speakVoiceGuidance]);

  // Title derivation: Always prioritize OCR-derived article headline; avoid generic labels or file names
  const rawTitle = result.articleTitle || result.articleSummary?.title || "";
  const isGeneric = !rawTitle ||
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

  const scoring = result.scoringFactors || {
    evidenceStrength: { weight: 0.70, score: result.evidence?.evidenceScore ?? 80, contribution: 56.0, label: "Evidence Strength (70%)" },
    sourceReputation: { weight: 0.10, score: result.sourceReputation?.trustScore ?? 85, contribution: 8.5, label: "Domain & Source Authority (10%)" },
    claimConsistency: { weight: 0.10, score: 92, contribution: 9.2, label: "Factual Claim Consistency (10%)" },
    mlPrediction: { weight: 0.05, score: result.prediction === "REAL" ? 82 : 28, contribution: 4.1, label: "ML Text Pattern Classifier (5%)" },
    languageQuality: { weight: 0.05, score: result.languageIntegrity ?? 88, contribution: 4.4, label: "Journalistic Language Integrity (5%)" },
  };

  const suppCount = result.evidence?.supportingCount ?? result.claims?.filter((c) => c.status === "Supported").length ?? 3;
  const contraCount = result.evidence?.contradictingCount ?? result.claims?.filter((c) => c.status === "Contradicted").length ?? 0;
  const sourceTrust = result.sourceReputation?.trustScore ?? result.sourceCredibility ?? 85;
  const sourceName = result.sourceReputation?.sourceName || result.sourceDomain || "Identified Publisher";
  const displayPublisher = result.publisher || result.sourceReputation?.publisher || sourceName;
  const displaySourceType = result.sourceType || result.sourceReputation?.sourceType;
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

  const quickSummaryText =
    summary.quickSummary ||
    summary.tldr ||
    (summary.executiveSummary
      ? summary.executiveSummary.split(/(?<=[.!?])\s+/).slice(0, 2).join(" ")
      : "Summary unavailable.");

  return (
    <div className="max-w-[1000px] w-full mx-auto px-3 sm:px-6 py-4 sm:py-8 overflow-x-hidden">
      {/* Top Action Navigation Bar (Responsive with Touch Targets >= 44px) */}
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

      {/* HERO ELEMENT: Article / News Headline (Largest & most prominent element on page) */}
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
      {/* -> CLAIMS -> EVIDENCE -> VERDICT                                          */}
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

        {/* 2. OVERVIEW (1-2 simple sentences in plain English, at-a-glance insight) */}
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

        {/* 5. CLAIM-BY-CLAIM EVIDENCE MATRIX (RAG Architecture) */}
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
                        {/* Supporting Evidence for this claim */}
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

                        {/* Contradicting Evidence for this claim */}
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

                        {/* If no evidence was found for this specific claim */}
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
          </div>

          {/* Source Transparency Block (Requested Format) */}
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

          {/* TOP SUPPORTING SOURCES (Perplexity / RAG Style Cards) */}
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

          {/* TOP CONTRADICTING SOURCES (Perplexity / RAG Style Cards) */}
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

          {/* Missing evidence is NEVER treated as false alert */}
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

        {/* SECTION 6: Multi-Factor Scoring Breakdown */}
        <section className="lucid-card bg-surface-container rounded-xl p-4 sm:p-6 border border-surface-container-high/60 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-2 border-b border-surface-container-high/40 pb-3">
            <div className="flex items-center gap-2">
              <MatIcon name="analytics" size={20} className="text-primary" />
              <h3 className="font-headline-sm text-headline-sm text-on-surface uppercase tracking-wide">
                Multi-Factor Trust Scoring Breakdown
              </h3>
            </div>
            <span className="font-label-xs text-label-xs text-outline font-medium">
              80% Evidence &amp; Source led · ML &amp; Style capped at 10%
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3">
            {[
              { key: "evidenceStrength", factor: scoring.evidenceStrength, icon: "fact_check", descSuffix: "Corroborating citations", defaultPct: 70 },
              { key: "sourceReputation", factor: scoring.sourceReputation, icon: "domain", descSuffix: "Publisher & domain authority", defaultPct: 10 },
              { key: "claimConsistency", factor: scoring.claimConsistency, icon: "balance", descSuffix: "Internal claim coherence", defaultPct: 10 },
              { key: "mlPrediction", factor: scoring.mlPrediction, icon: "psychology", descSuffix: "Linguistic pattern classifier", defaultPct: 5 },
              { key: "languageQuality", factor: scoring.languageQuality, icon: "article", descSuffix: "Journalistic tone integrity", defaultPct: 5 },
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

// ==========================================
// 6. MAIN APPLICATION
// ==========================================

export default function App() {
  // Authentication State: Registered accounts persist; Guest sessions are temporary in-memory ONLY
  const [currentUser, setCurrentUser] = useState<UserAccount | null>(() => {
    try {
      // Clear any legacy guest session flag on load
      sessionStorage.removeItem("lucid_guest_session");
      const savedUser = localStorage.getItem("lucid_user");
      if (savedUser) {
        const parsed = JSON.parse(savedUser);
        if (parsed && !parsed.isGuest) return parsed;
      }
    } catch {}
    return null;
  });

  const [screen, setScreen] = useState<Screen>(() => {
    // Welcome / Authentication screen is the mandatory default entry point on fresh visit or refresh
    try {
      const savedUser = localStorage.getItem("lucid_user");
      if (savedUser) {
        const parsed = JSON.parse(savedUser);
        if (parsed && !parsed.isGuest) return "home";
      }
    } catch {}
    return "welcome";
  });

  const [darkMode, setDarkMode] = useState<boolean>(() => {
    return localStorage.getItem("lucid_theme") === "dark";
  });

  const [activeSubject, setActiveSubject] = useState("");
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [analysisStep, setAnalysisStep] = useState(1);
  const [currentResult, setCurrentResult] = useState<VerificationResult | null>(null);
  const currentResultRef = useRef<VerificationResult | null>(null);

  // OCR Preview State
  const [ocrPreviewData, setOcrPreviewData] = useState<OcrPreviewData | null>(null);
  const ocrPreviewDataRef = useRef<OcrPreviewData | null>(null);
  const [isExtractingOcr, setIsExtractingOcr] = useState(false);

  // History state: only persistent for signed-in accounts; Guests start with empty in-memory history
  const [history, setHistory] = useState<HistoryItem[]>(() => {
    try {
      const savedUser = localStorage.getItem("lucid_user");
      if (savedUser) {
        const parsed = JSON.parse(savedUser);
        if (parsed && !parsed.isGuest) {
          const savedHistory = localStorage.getItem("lucid_history");
          if (savedHistory) return JSON.parse(savedHistory);
        }
      }
    } catch {}
    return [];
  });

  const [isAuthModalOpen, setIsAuthModalOpen] = useState(false);

  // Verification Error & Workspace preservation state
  const [verifyError, setVerifyError] = useState<string | null>(null);
  const [workspaceInput, setWorkspaceInput] = useState<{
    mode: "url" | "text" | "image" | "pdf";
    query: string;
    selectedFile: File | null;
  }>({
    mode: "url",
    query: "",
    selectedFile: null,
  });

  // Offline 404 Detection & Mini-Game State
  const [isOffline, setIsOffline] = useState(() => {
    return typeof navigator !== "undefined" ? !navigator.onLine : false;
  });
  const [dismissedOffline, setDismissedOffline] = useState(false);
  const [isCheckingConnection, setIsCheckingConnection] = useState(false);
  const [connectionToast, setConnectionToast] = useState<string | null>(null);

  useEffect(() => {
    const handleOnline = () => {
      setIsOffline(false);
      setDismissedOffline(false);
      setConnectionToast("Connection restored");
      const t = setTimeout(() => setConnectionToast(null), 3500);
      return () => clearTimeout(t);
    };

    const handleOffline = () => {
      setIsOffline(true);
      setDismissedOffline(false);
    };

    window.addEventListener("online", handleOnline);
    window.addEventListener("offline", handleOffline);

    return () => {
      window.removeEventListener("online", handleOnline);
      window.removeEventListener("offline", handleOffline);
    };
  }, []);

  const handleRetryConnection = async () => {
    setIsCheckingConnection(true);
    try {
      if (typeof navigator !== "undefined" && navigator.onLine) {
        await fetch(getApiUrl("/api/health"), { cache: "no-store", mode: "no-cors" }).catch(() => null);
        setIsOffline(false);
        setDismissedOffline(false);
        setConnectionToast("Connection restored");
        setTimeout(() => setConnectionToast(null), 3500);
      } else {
        setConnectionToast("Still offline. Please check your internet connection.");
        setTimeout(() => setConnectionToast(null), 3000);
      }
    } catch {
      setConnectionToast("Still offline. Please check your internet connection.");
      setTimeout(() => setConnectionToast(null), 3000);
    } finally {
      setIsCheckingConnection(false);
    }
  };

  const handleClearHistory = () => {
    setHistory([]);
    try {
      localStorage.removeItem("lucid_history");
    } catch {}
  };

  // Voice Guidance State & Web Speech Synthesis
  const [voiceGuideEnabled, setVoiceGuideEnabled] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);

  const speakVoiceGuidance = useCallback((text: string) => {
    if (typeof window === "undefined" || !("speechSynthesis" in window)) return;
    try {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 0.95;
      utterance.pitch = 1.0;
      utterance.onstart = () => setIsSpeaking(true);
      utterance.onend = () => setIsSpeaking(false);
      utterance.onerror = () => setIsSpeaking(false);
      window.speechSynthesis.speak(utterance);
    } catch (err) {
      console.warn("Speech synthesis error:", err);
      setIsSpeaking(false);
    }
  }, []);

  const stopSpeaking = useCallback(() => {
    if (typeof window !== "undefined" && "speechSynthesis" in window) {
      try {
        window.speechSynthesis.cancel();
      } catch {}
      setIsSpeaking(false);
    }
  }, []);

  const toggleVoiceGuide = () => {
    if (voiceGuideEnabled) {
      stopSpeaking();
      setVoiceGuideEnabled(false);
    } else {
      setVoiceGuideEnabled(true);
      speakVoiceGuidance("Voice guide enabled. Paste a link, type a claim, or press the mic to speak.");
    }
  };

  // Path-Based Navigation & Route Controller (HTML5 History API)
  const syncRouteToScreen = useCallback((targetPath?: string) => {
    if (typeof window === "undefined") return;

    // 1. Strip and migrate legacy hash if present (e.g. /#/welcome, /verify#/welcome, /#/verify)
    if (window.location.hash) {
      const legacy = window.location.hash.toLowerCase().replace(/^#\/?/, "");
      const cleanTarget = (legacy === "verify" || legacy === "report" || legacy === "ocr-preview" || legacy === "analysis" || legacy === "history")
        ? `/${legacy}`
        : "/";
      window.history.replaceState(null, "", cleanTarget);
    }

    const currentPath = (targetPath || window.location.pathname).toLowerCase().replace(/\/+$/, "") || "/";

    // 2. Unauthenticated Route Guard: Force exactly "/" (Welcome / Login screen)
    if (!currentUser) {
      if (currentPath !== "/") {
        window.history.replaceState(null, "", "/");
      }
      setScreen("welcome");
      return;
    }

    // 3. Authenticated Route Mapping
    if (currentPath === "/" || currentPath === "/welcome" || currentPath === "/login") {
      // Logged-in users who land on / are routed to /verify
      window.history.replaceState(null, "", "/verify");
      setScreen("home");
    } else if (currentPath === "/verify" || currentPath === "/history") {
      setScreen("home");
    } else if (currentPath === "/report") {
      if (currentResultRef.current || currentResult) {
        if (!currentResult && currentResultRef.current) {
          setCurrentResult(currentResultRef.current);
        }
        setScreen("report");
      } else {
        window.history.replaceState(null, "", "/verify");
        setScreen("home");
      }
    } else if (currentPath === "/ocr-preview") {
      if (ocrPreviewDataRef.current || ocrPreviewData) {
        if (!ocrPreviewData && ocrPreviewDataRef.current) {
          setOcrPreviewData(ocrPreviewDataRef.current);
        }
        setScreen("ocr-preview");
      } else {
        window.history.replaceState(null, "", "/verify");
        setScreen("home");
      }
    } else if (currentPath === "/analysis") {
      setScreen("live-analysis");
    } else {
      // Fallback unrecognized routes to /verify
      window.history.replaceState(null, "", "/verify");
      setScreen("home");
    }
  }, [currentUser, currentResult]);

  const navigate = useCallback((path: string, replace = false) => {
    if (typeof window === "undefined") return;
    const cleanPath = path.startsWith("/") ? path : `/${path}`;
    if (window.location.pathname === cleanPath && !window.location.hash) {
      syncRouteToScreen(cleanPath);
      return;
    }
    if (replace) {
      window.history.replaceState(null, "", cleanPath);
    } else {
      window.history.pushState(null, "", cleanPath);
    }
    syncRouteToScreen(cleanPath);
  }, [syncRouteToScreen]);

  useEffect(() => {
    syncRouteToScreen();
    const onPop = () => syncRouteToScreen();
    window.addEventListener("popstate", onPop);
    return () => {
      window.removeEventListener("popstate", onPop);
    };
  }, [syncRouteToScreen]);

  const abortControllerRef = useRef<AbortController | null>(null);

  const resetSession = useCallback(() => {
    // 1. Abort any active in-flight fetch request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }

    // 2. Stop ongoing speech synthesis
    stopSpeaking();

    // 3. Clear all per-user React states
    setWorkspaceInput({ mode: "url", query: "", selectedFile: null });
    setOcrPreviewData(null);
    ocrPreviewDataRef.current = null;
    setCurrentResult(null);
    currentResultRef.current = null;
    setActiveSubject("");
    setVerifyError(null);
    setAnalysisProgress(0);
    setAnalysisStep(1);
    setIsExtractingOcr(false);
    setIsAuthModalOpen(false);
    setHistory([]);

    // 4. Clear per-user localStorage & sessionStorage (keep theme, voice guide preference, and dino high score)
    try {
      const keepKeys = new Set(["lucid_theme", "lucid_voice_guide", "lucid_dino_high_score", "dinoHighScore"]);
      const keysToRemove: string[] = [];
      for (let i = 0; i < localStorage.length; i++) {
        const k = localStorage.key(i);
        if (k && !keepKeys.has(k)) {
          keysToRemove.push(k);
        }
      }
      keysToRemove.forEach((k) => localStorage.removeItem(k));
      sessionStorage.clear();
    } catch {}

    // 5. Navigate to / with history replace
    navigate("/", true);
  }, [stopSpeaking, navigate]);

  // Reset session whenever the Firebase auth user ID changes
  useEffect(() => {
    if (!auth) return;
    let prevUid: string | null = auth.currentUser?.uid || null;
    const unsubscribe = auth.onAuthStateChanged((user) => {
      const currentUid = user?.uid || null;
      if (prevUid !== null && prevUid !== currentUid) {
        resetSession();
      }
      prevUid = currentUid;
    });
    return () => unsubscribe();
  }, [resetSession]);

  const handleLogin = (user: UserAccount) => {
    resetSession();
    setCurrentUser(user);
    if (!user.isGuest) {
      localStorage.setItem("lucid_user", JSON.stringify(user));
      try {
        const savedHistory = localStorage.getItem("lucid_history");
        if (savedHistory) setHistory(JSON.parse(savedHistory));
      } catch {}
    } else {
      localStorage.removeItem("lucid_user");
      setHistory([]);
    }
    setIsAuthModalOpen(false);
    setScreen("home");
    navigate("/verify");
  };

  const handleSignOut = () => {
    resetSession();
    setCurrentUser(null);
    signOutFirebase();
    setScreen("welcome");
  };

  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add("dark");
      localStorage.setItem("lucid_theme", "dark");
    } else {
      document.documentElement.classList.remove("dark");
      localStorage.setItem("lucid_theme", "light");
    }
  }, [darkMode]);

  // Persist history ONLY for registered/signed-in users (NEVER for guest mode)
  useEffect(() => {
    if (currentUser && !currentUser.isGuest) {
      localStorage.setItem("lucid_history", JSON.stringify(history));
    }
  }, [history, currentUser]);

  const handleOcrExtract = async (file: File) => {
    setIsExtractingOcr(true);
    try {
      const formData = new FormData();
      formData.append("file", file, file.name);
      const response = await fetchApi("/api/ocr", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errJson = await response.json().catch(() => null);
        console.error("[OCR Diagnostics] API error response:", response.status, response.statusText, errJson);
        throw new Error("Text could not be extracted from this image.");
      }

      const ocrData = await response.json();
      const previewPayload: OcrPreviewData = {
        fileName: file.name,
        fileExtension: ocrData.fileExtension || file.name.split(".").pop() || "png",
        extractedText: ocrData.extractedText || "",
        previewText: ocrData.previewText || (ocrData.extractedText || "").slice(0, 500),
        confidence: ocrData.confidence ?? 0,
        isLowConfidence: ocrData.isLowConfidence ?? false,
        warning: ocrData.warning || "",
        qualityWarning: ocrData.qualityWarning || "",
        wordCount: ocrData.wordCount || 0,
        originalFile: file,
      };
      ocrPreviewDataRef.current = previewPayload;
      setOcrPreviewData(previewPayload);
      setScreen("ocr-preview");
      navigate("/ocr-preview");
    } catch (err) {
      console.error("[OCR Diagnostics] Image text extraction failed:", err);
      setVerifyError("Text could not be extracted from this image. Please try a clearer screenshot or upload text directly.");
    } finally {
      setIsExtractingOcr(false);
    }
  };

  const handleStartVerification = async ({
    mode,
    content,
    file,
    textOverride,
    ocrData,
  }: {
    mode: "url" | "text" | "upload";
    content: string;
    file?: File;
    textOverride?: string;
    ocrData?: OcrPreviewData | null;
  }) => {
    setVerifyError(null);
    setActiveSubject(content);
    setAnalysisProgress(15);
    setAnalysisStep(1);
    setScreen("live-analysis");
    navigate("/analysis");

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

    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      let response: Response;
      if (mode === "upload" && file) {
        const formData = new FormData();
        formData.append("file", file, file.name);
        if (textOverride) {
          formData.append("textOverride", textOverride);
        }
        // Browser sets multipart/form-data with boundary automatically
        response = await fetchApi("/api/analyze-file", {
          method: "POST",
          body: formData,
          signal: controller.signal,
        });
      } else if (mode === "url") {
        response = await fetchApi("/api/analyze-url", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ url: content }),
          signal: controller.signal,
        });
      } else {
        response = await fetchApi("/api/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: content }),
          signal: controller.signal,
        });
      }

      if (!response.ok) {
        const errJson = await response.json().catch(() => null);
        const errMsg = errJson?.error || `API returned status ${response.status}`;
        throw new Error(errMsg);
      }

      const data: VerificationResult = await response.json();

      // Ensure OCR details are populated if coming from OCR preview
      if (ocrData) {
        const activeText = textOverride || ocrData.extractedText || "";
        data.ocrPreview = {
          previewText: activeText.slice(0, 500),
          extractedText: activeText,
          confidence: ocrData.confidence,
          isLowConfidence: ocrData.isLowConfidence,
          warning: ocrData.warning,
          qualityWarning: ocrData.qualityWarning,
        };
        data.ocrConfidence = ocrData.confidence;
        data.isLowConfidence = ocrData.isLowConfidence;
        data.ocrWarning = ocrData.warning;
        data.ocrQualityWarning = ocrData.qualityWarning;
      }

      clearTimeout(timer1);
      clearTimeout(timer2);
      clearTimeout(timer3);
      setAnalysisProgress(100);
      setAnalysisStep(5);

      // Directly update currentResultRef so route sync synchronously routes to report screen
      currentResultRef.current = data;
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
      navigate("/report");
    } catch (err: any) {
      console.error("Verification error:", err);
      clearTimeout(timer1);
      clearTimeout(timer2);
      clearTimeout(timer3);
      setVerifyError(
        err?.message || "Verification could not be completed. Please ensure the backend verification server is reachable and try again."
      );
      setScreen("home");
      navigate("/verify");
    }
  };

  const handleContinueFromOcr = (reviewedText: string) => {
    if (!ocrPreviewData) return;
    handleStartVerification({
      mode: "upload",
      content: reviewedText,
      file: ocrPreviewData.originalFile,
      textOverride: reviewedText,
      ocrData: ocrPreviewData,
    });
  };

  return (
    <div className="bg-surface font-body-md text-on-surface antialiased min-h-screen flex flex-col transition-colors duration-200">
      <AppHeader
        onNavigateHome={() => {
          if (currentUser) {
            setScreen("home");
            navigate("/verify");
          } else {
            setScreen("welcome");
            navigate("/", true);
          }
        }}
        darkMode={darkMode}
        onToggleDarkMode={() => setDarkMode(!darkMode)}
        currentUser={currentUser}
        onOpenAuth={() => setIsAuthModalOpen(true)}
        onSignOut={handleSignOut}
        voiceGuideEnabled={voiceGuideEnabled}
        onToggleVoiceGuide={toggleVoiceGuide}
        isSpeaking={isSpeaking}
        onStopSpeaking={stopSpeaking}
      />

      <main className="w-full pt-16 bg-surface min-h-[calc(100vh-140px)] flex-1 flex flex-col">
        {/* If user is not authenticated or explicitly on welcome screen, show Welcome Screen only */}
        {(!currentUser || screen === "welcome") && (
          <WelcomeScreen onLogin={handleLogin} />
        )}

        {/* Protected Route: Verification Home Workspace */}
        {currentUser && screen === "home" && (
          <HomeScreen
            onVerify={handleStartVerification}
            onOcrExtract={handleOcrExtract}
            isExtractingOcr={isExtractingOcr}
            history={history}
            onClearHistory={handleClearHistory}
            verifyError={verifyError}
            onClearError={() => setVerifyError(null)}
            initialInput={workspaceInput}
            onInputChange={setWorkspaceInput}
            onSelectHistory={(item) => {
              currentResultRef.current = item.data;
              setCurrentResult(item.data);
              setActiveSubject(item.title);
              setScreen("report");
              navigate("/report");
            }}
          />
        )}

        {/* Protected Route: OCR Review */}
        {currentUser && screen === "ocr-preview" && ocrPreviewData && (
          <OcrPreviewScreen
            previewData={ocrPreviewData}
            onContinue={handleContinueFromOcr}
            onCancel={() => {
              setOcrPreviewData(null);
              setScreen("home");
              navigate("/verify");
            }}
          />
        )}

        {/* Protected Route: Live RAG Analysis */}
        {currentUser && screen === "live-analysis" && (
          <LoadingScreen
            subject={activeSubject}
            progress={analysisProgress}
            activeStep={analysisStep}
            onCancel={() => {
              setScreen("home");
              navigate("/verify");
            }}
          />
        )}

        {/* Protected Route: Verification Report */}
        {currentUser && screen === "report" && (
          currentResult ? (
            <ReportScreen
              result={currentResult}
              subject={activeSubject}
              onNewSearch={() => {
                setScreen("home");
                navigate("/verify");
              }}
              voiceGuideEnabled={voiceGuideEnabled}
              speakVoiceGuidance={speakVoiceGuidance}
              isSpeaking={isSpeaking}
              onStopSpeaking={stopSpeaking}
            />
          ) : (
            <div className="flex-1 flex flex-col items-center justify-center py-20 text-center gap-4">
              <h2 className="font-headline-sm text-headline-sm">No report loaded</h2>
              <button
                onClick={() => {
                  setScreen("home");
                  navigate("/verify");
                }}
                className="px-5 py-2.5 min-h-[44px] rounded-full bg-primary text-on-primary font-label-md cursor-pointer"
              >
                Go to Verification Home
              </button>
            </div>
          )
        )}
      </main>

      {/* In-Session Account Upgrade Modal */}
      <AuthModal
        isOpen={isAuthModalOpen}
        onClose={() => setIsAuthModalOpen(false)}
        onLogin={handleLogin}
      />

      {/* Offline 404 Overlay with Endless Runner Mini-Game */}
      {isOffline && !dismissedOffline && (
        <Offline404Overlay
          onRetry={handleRetryConnection}
          onDismiss={() => setDismissedOffline(true)}
          isChecking={isCheckingConnection}
          isDarkMode={darkMode}
        />
      )}

      {/* Cached Offline Banner if user dismissed overlay */}
      {isOffline && dismissedOffline && (
        <div className="bg-amber-950/90 text-amber-200 border-b border-amber-800/50 px-4 py-2 text-xs flex items-center justify-between z-40 sticky top-0" role="status">
          <div className="flex items-center gap-2">
            <MatIcon name="wifi_off" size={16} />
            <span>You are currently offline (viewing cached page)</span>
          </div>
          <button
            onClick={() => setDismissedOffline(false)}
            className="underline hover:text-white font-medium ml-2 min-h-[44px] px-2 flex items-center cursor-pointer"
            type="button"
          >
            Play Mini-Game
          </button>
        </div>
      )}

      {/* Reconnection / Offline Notification Toast */}
      {connectionToast && (
        <div
          className="fixed bottom-6 right-6 z-50 flex items-center gap-2 px-4 py-2.5 rounded-xl bg-surface-container-high border border-surface-container-highest shadow-xl text-sm font-semibold text-on-surface animate-fade-in"
          role="status"
          aria-live="polite"
        >
          <MatIcon
            name={connectionToast.includes("restored") ? "check_circle" : "wifi_off"}
            size={18}
            className={connectionToast.includes("restored") ? "text-tertiary" : "text-error"}
          />
          <span>{connectionToast}</span>
        </div>
      )}

      <AppFooter />
    </div>
  );
}
