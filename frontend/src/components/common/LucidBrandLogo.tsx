import React from "react";

export function LucidBrandLogo({
  size = "md",
  showText = true,
  onClick,
  className = "",
  imgClassName = "",
}: {
  size?: "sm" | "md" | "lg" | "hero";
  showText?: boolean;
  onClick?: () => void;
  className?: string;
  imgClassName?: string;
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
      className={`flex items-center gap-2.5 select-none ${onClick ? "cursor-pointer" : ""} ${className}`}
    >
      <img
        src="/logo-transparent.png"
        alt="Lucid Logo"
        className={`${heightClasses[size]} w-auto object-contain transition-transform duration-200 ${imgClassName}`}
      />
      {showText && (
        <span className="font-headline-sm text-headline-sm tracking-tight text-on-surface font-bold">
          Lucid
        </span>
      )}
    </div>
  );
}
