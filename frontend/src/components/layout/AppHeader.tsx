import React, { useState } from "react";
import { MatIcon } from "../common/MatIcon";
import { LucidBrandLogo } from "../common/LucidBrandLogo";
import { UserAccount } from "../../types/verification";

export function AppHeader({
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
