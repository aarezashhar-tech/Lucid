import { initializeApp, getApps, getApp, FirebaseApp } from "firebase/app";
import {
  getAuth,
  GoogleAuthProvider,
  OAuthProvider,
  signInWithPopup,
  signOut as fbSignOut,
  Auth,
} from "firebase/auth";

// Firebase configuration read from Vite environment variables.
// These variables must be provided in Vercel or local .env (never commit secrets to repo).
const firebaseConfig = {
  apiKey: import.meta.env.VITE_FIREBASE_API_KEY || "",
  authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN || "",
  projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID || "",
  storageBucket: import.meta.env.VITE_FIREBASE_STORAGE_BUCKET || "",
  messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID || "",
  appId: import.meta.env.VITE_FIREBASE_APP_ID || "",
};

export const isFirebaseConfigured: boolean = Boolean(
  firebaseConfig.apiKey &&
    firebaseConfig.authDomain &&
    firebaseConfig.projectId
);

let app: FirebaseApp | null = null;
let authInstance: Auth | null = null;

if (isFirebaseConfigured) {
  try {
    app = getApps().length > 0 ? getApp() : initializeApp(firebaseConfig);
    authInstance = getAuth(app);
  } catch (error) {
    console.error("Failed to initialize Firebase Auth:", error);
  }
}

export const auth = authInstance;

export const googleProvider = new GoogleAuthProvider();
googleProvider.setCustomParameters({ prompt: "select_account" });

// Microsoft OAuth Provider using Microsoft Entra / Microsoft Account
export const microsoftProvider = new OAuthProvider("microsoft.com");
microsoftProvider.setCustomParameters({ prompt: "select_account" });
microsoftProvider.addScope("User.Read");

export interface AuthSuccessUser {
  name: string;
  email: string;
  isGuest: boolean;
  avatar?: string;
}

export interface AuthResult {
  user?: AuthSuccessUser;
  error?: string;
}

/**
 * Initiates Google OAuth Popup via Firebase Auth
 */
export async function signInWithGoogle(): Promise<AuthResult> {
  if (!isFirebaseConfigured || !auth) {
    return {
      error:
        "OAuth is not configured. Please set the VITE_FIREBASE_* environment variables in Vercel or .env.",
    };
  }

  try {
    const result = await signInWithPopup(auth, googleProvider);
    const fbUser = result.user;
    return {
      user: {
        name: fbUser.displayName || fbUser.email?.split("@")[0] || "Google User",
        email: fbUser.email || "",
        isGuest: false,
        avatar: fbUser.photoURL || undefined,
      },
    };
  } catch (err: any) {
    if (err.code === "auth/popup-closed-by-user") {
      return { error: "Google sign-in popup was closed before completing." };
    }
    if (err.code === "auth/popup-blocked") {
      return { error: "Sign-in popup was blocked by browser. Please allow popups for this site." };
    }
    if (err.code === "auth/unauthorized-domain") {
      return {
        error:
          "This domain is not authorized in Firebase Console. Add your Vercel or local URL to Firebase Auth > Settings > Authorized Domains.",
      };
    }
    return { error: err.message || "Failed to sign in with Google." };
  }
}

/**
 * Initiates Microsoft OAuth Popup via Firebase Auth
 */
export async function signInWithMicrosoft(): Promise<AuthResult> {
  if (!isFirebaseConfigured || !auth) {
    return {
      error:
        "OAuth is not configured. Please set the VITE_FIREBASE_* environment variables in Vercel or .env.",
    };
  }

  try {
    const result = await signInWithPopup(auth, microsoftProvider);
    const fbUser = result.user;
    return {
      user: {
        name: fbUser.displayName || fbUser.email?.split("@")[0] || "Microsoft User",
        email: fbUser.email || "",
        isGuest: false,
        avatar: fbUser.photoURL || undefined,
      },
    };
  } catch (err: any) {
    if (err.code === "auth/popup-closed-by-user") {
      return { error: "Microsoft sign-in popup was closed before completing." };
    }
    if (err.code === "auth/popup-blocked") {
      return { error: "Sign-in popup was blocked by browser. Please allow popups for this site." };
    }
    if (err.code === "auth/unauthorized-domain") {
      return {
        error:
          "This domain is not authorized in Firebase Console. Add your Vercel or local URL to Firebase Auth > Settings > Authorized Domains.",
      };
    }
    return { error: err.message || "Failed to sign in with Microsoft." };
  }
}

/**
 * Sign out from Firebase if an active session exists
 */
export async function signOutFirebase(): Promise<void> {
  if (auth) {
    try {
      await fbSignOut(auth);
    } catch (err) {
      console.warn("Firebase sign out error:", err);
    }
  }
}
