import React, { useState, useEffect, useRef, useCallback } from "react";
import {
  Screen,
  UserAccount,
  VerificationResult,
  OcrPreviewData,
  HistoryItem,
} from "./types/verification";
import { getApiUrl, fetchApi } from "./services/api";
import { auth, signOutFirebase } from "./firebase";

// Common & Layout Components
import { MatIcon } from "./components/common/MatIcon";
import { AppHeader } from "./components/layout/AppHeader";
import { AppFooter } from "./components/layout/AppFooter";

// Modals
import { AuthModal } from "./components/modals/AuthModal";
import { Offline404Overlay } from "./components/modals/Offline404Overlay";

// Screen Components
import { WelcomeScreen } from "./components/screens/WelcomeScreen";
import { HomeScreen } from "./components/screens/HomeScreen";
import { OcrPreviewScreen } from "./components/screens/OcrPreviewScreen";
import { LoadingScreen } from "./components/screens/LoadingScreen";
import { ReportScreen } from "./components/screens/ReportScreen";

export default function App() {
  // Authentication State: Registered accounts persist; Guest sessions are temporary in-memory ONLY
  const [currentUser, setCurrentUser] = useState<UserAccount | null>(() => {
    try {
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

    if (window.location.hash) {
      const legacy = window.location.hash.toLowerCase().replace(/^#\/?/, "");
      const cleanTarget =
        legacy === "verify" || legacy === "report" || legacy === "ocr-preview" || legacy === "analysis" || legacy === "history"
          ? `/${legacy}`
          : "/";
      window.history.replaceState(null, "", cleanTarget);
    }

    const currentPath = (targetPath || window.location.pathname).toLowerCase().replace(/\/+$/, "") || "/";

    if (!currentUser) {
      if (currentPath !== "/") {
        window.history.replaceState(null, "", "/");
      }
      setScreen("welcome");
      return;
    }

    if (currentPath === "/" || currentPath === "/welcome" || currentPath === "/login") {
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
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }

    stopSpeaking();

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

    navigate("/", true);
  }, [stopSpeaking, navigate]);

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
        const errMsg = errJson?.error || "Text could not be extracted from this image.";
        throw new Error(errMsg);
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
    } catch (err: any) {
      console.error("[OCR Diagnostics] Image text extraction failed:", err);
      setVerifyError(
        err?.message || "Text could not be extracted from this image. Please try a clearer screenshot or upload text directly."
      );
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
        {/* Unauthenticated / Welcome Screen */}
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

        {/* Protected Route: Live RAG Analysis Stepper */}
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
