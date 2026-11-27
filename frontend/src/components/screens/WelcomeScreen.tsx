import React, { useState } from "react";
import { MatIcon } from "../common/MatIcon";
import { LucidBrandLogo } from "../common/LucidBrandLogo";
import { signInWithGoogle, isFirebaseConfigured } from "../../firebase";
import { handleCardPointerMove } from "../../services/api";
import { UserAccount } from "../../types/verification";

export function WelcomeScreen({
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

  const handleGuestLogin = () => {
    onLogin({
      name: "Guest Investigator",
      email: "guest@lucid.local",
      isGuest: true,
    });
  };

  return (
    <div className="w-full flex-1 flex flex-col items-center justify-center px-4 py-8 sm:py-16 animate-fade-in">
      <div
        onPointerMove={handleCardPointerMove}
        className="w-full max-w-[420px] lucid-card glass-card spotlight-card rounded-2xl shadow-xl p-6 sm:p-8 flex flex-col items-center gap-5 text-center transition-all border border-surface-container-high/60 bg-surface-container-lowest"
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
            className="w-full min-h-[46px] flex items-center justify-center gap-3 px-4 py-2.5 rounded-xl border border-surface-container-high hover:bg-surface-container-low transition-all font-label-md text-label-md text-on-surface font-semibold cursor-pointer disabled:opacity-60 disabled:cursor-not-allowed shadow-xs active:scale-[0.98]"
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
            <span>Continue with Google</span>
          </button>
        </div>

        {/* Divider */}
        <div className="w-full flex items-center gap-3 my-1">
          <div className="flex-1 h-px bg-surface-container-high" />
          <span className="text-xs text-outline uppercase tracking-wider font-medium">Or</span>
          <div className="flex-1 h-px bg-surface-container-high" />
        </div>

        {/* One-click Guest Demo Access */}
        <button
          onClick={handleGuestLogin}
          className="w-full min-h-[44px] flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl bg-surface-container hover:bg-surface-container-high text-on-surface font-label-md text-label-md font-medium transition-all cursor-pointer shadow-xs active:scale-[0.98]"
          type="button"
        >
          <MatIcon name="explore" size={18} className="text-primary" />
          <span>Explore Instantly as Guest</span>
        </button>
      </div>
    </div>
  );
}
