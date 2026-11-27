import React, { useState } from "react";
import { MatIcon } from "../common/MatIcon";
import { LucidBrandLogo } from "../common/LucidBrandLogo";
import { UserAccount } from "../../types/verification";
import { isFirebaseConfigured, signInWithGoogle } from "../../firebase";

export function AuthModal({
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
            id="modal-email-input"
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
