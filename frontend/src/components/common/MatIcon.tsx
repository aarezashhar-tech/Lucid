import React from "react";

export function MatIcon({
  name,
  className = "",
  size = 20,
  fill = false,
}: {
  name: string;
  className?: string;
  size?: number;
  fill?: boolean;
}) {
  return (
    <span
      className={`material-symbols-outlined select-none leading-none ${className}`}
      style={{
        fontSize: `${size}px`,
        fontVariationSettings: fill ? "'FILL' 1" : "'FILL' 0",
      }}
    >
      {name}
    </span>
  );
}
