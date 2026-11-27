import React, { useState, useRef, useEffect, FormEvent } from "react";
import { MatIcon } from "../common/MatIcon";
import { LucidBrandLogo } from "../common/LucidBrandLogo";
import { HistoryItem } from "../../types/verification";
import { handleCardPointerMove } from "../../services/api";

export function HomeScreen({
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
