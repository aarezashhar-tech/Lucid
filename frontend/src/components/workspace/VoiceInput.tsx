import React, { useState, useRef, useEffect } from "react";
import { MatIcon } from "../common/MatIcon";

export function VoiceInput({
  onTranscript,
  className = "",
}: {
  onTranscript: (text: string) => void;
  className?: string;
}) {
  const [isListening, setIsListening] = useState(false);
  const [speechSupported, setSpeechSupported] = useState(true);
  const [voiceLanguage, setVoiceLanguage] = useState<"en-IN" | "ta-IN" | "hi-IN">("en-IN");
  const [voiceError, setVoiceError] = useState<string | null>(null);
  const [showLangMenu, setShowLangMenu] = useState(false);
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
          onTranscript(liveTranscript);
        }
      };

      recognition.onerror = (event: any) => {
        setIsListening(false);
        if (event.error === "not-allowed" || event.error === "service-not-allowed") {
          setVoiceError("Microphone access denied. Please allow microphone permissions in your browser.");
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
      console.error("Speech recognition error:", err);
      setIsListening(false);
      setVoiceError("Could not initialize voice input on this browser.");
    }
  };

  if (!speechSupported) return null;

  const languages = [
    { code: "en-IN" as const, label: "English (India)" },
    { code: "ta-IN" as const, label: "Tamil (தமிழ்)" },
    { code: "hi-IN" as const, label: "Hindi (हिन्दी)" },
  ];

  return (
    <div className={`relative flex items-center gap-1.5 ${className}`}>
      {/* Mic Button */}
      <button
        onClick={toggleVoiceInput}
        type="button"
        title={isListening ? "Listening... Click to stop" : "Voice Input (Speech to text)"}
        className={`p-2 rounded-full transition-all flex items-center justify-center cursor-pointer ${
          isListening
            ? "bg-rose-500 text-white animate-pulse shadow-md"
            : "text-on-surface-variant hover:text-primary hover:bg-surface-container"
        }`}
      >
        <MatIcon name={isListening ? "mic" : "mic_none"} size={20} fill={isListening} />
      </button>

      {/* Language Selector Dropdown Trigger */}
      <div className="relative">
        <button
          onClick={() => setShowLangMenu((p) => !p)}
          type="button"
          className="text-[11px] font-semibold font-mono uppercase tracking-wider text-on-surface-variant hover:text-primary px-1.5 py-0.5 rounded-md hover:bg-surface-container transition-colors cursor-pointer"
          title="Select Voice Language"
        >
          {voiceLanguage.split("-")[0]}
        </button>

        {showLangMenu && (
          <div className="absolute right-0 top-full mt-1.5 w-40 bg-surface-container-lowest border border-surface-container-high rounded-xl shadow-lg p-1.5 z-50 animate-scale-in">
            {languages.map((l) => (
              <button
                key={l.code}
                onClick={() => {
                  setVoiceLanguage(l.code);
                  setShowLangMenu(false);
                }}
                className={`w-full text-left px-2.5 py-1.5 text-xs rounded-lg transition-colors cursor-pointer flex items-center justify-between ${
                  voiceLanguage === l.code
                    ? "bg-primary/10 text-primary font-semibold"
                    : "text-on-surface-variant hover:bg-surface-container hover:text-on-surface"
                }`}
                type="button"
              >
                <span>{l.label}</span>
                {voiceLanguage === l.code && <MatIcon name="check" size={14} />}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Voice Error Tooltip */}
      {voiceError && (
        <div className="absolute top-full mt-2 left-0 right-0 p-2.5 rounded-xl bg-error-container text-on-error-container text-xs shadow-md z-40 flex items-start gap-1.5">
          <MatIcon name="error_outline" size={14} className="shrink-0 mt-0.5" />
          <span className="flex-1">{voiceError}</span>
          <button onClick={() => setVoiceError(null)} className="cursor-pointer text-outline hover:text-on-surface" type="button">
            <MatIcon name="close" size={12} />
          </button>
        </div>
      )}
    </div>
  );
}
