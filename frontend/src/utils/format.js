export function formatTime(value) {
  if (!value) return "";
  return new Intl.DateTimeFormat("fr-FR", {
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

export function pct(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${Math.round(number)}%` : "0%";
}

export function ms(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${Math.round(number)} ms` : "0 ms";
}

export function prob(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${Math.round(number * 1000) / 10}%` : "0%";
}

export function normalizeHorizon(value) {
  if (["3 jours", "3j", "3day", "bilstm_3d"].includes(value)) return "3 jours";
  return value || "unknown";
}
