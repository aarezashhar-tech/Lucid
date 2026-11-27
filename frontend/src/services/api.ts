import React from "react";
import { auth } from "../firebase";

/**
 * Resolves full API endpoint URL from import.meta.env.VITE_API_URL.
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
 * Lightweight mouse spotlight tracker using CSS variables --mx, --my.
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

export const cleanSnippetText = (str: string): string => {
  if (!str) return "";
  return str
    .replace(/<[^>]+>/g, " ")
    .replace(/https?:\/\/\S+/g, "")
    .replace(/["']?\s*<a\s+[^>]*>?/gi, "")
    .replace(/(target|href|class|rel)=["']?[^"'>\s]+["']?/gi, "")
    .replace(/\s+/g, " ")
    .trim();
};
