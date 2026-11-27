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
  executiveSummary: string;
  keyTakeaways: string[];
  tldr: string;
  quickSummary?: string;
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
  isDirectSubmission?: boolean;
  ragGroundedVerdict?: {
    verdict?: string;
    score?: number;
    confidence?: number;
    summary?: string;
  };
  overallVerdict?: string;
}

export interface HistoryItem {
  id: string;
  title: string;
  time: string;
  score: number | string;
  verdict: VerdictType;
  data: VerificationResult;
}

export interface UserAccount {
  name: string;
  email: string;
  isGuest: boolean;
  avatar?: string;
}
