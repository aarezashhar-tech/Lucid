"""
Article Summarization Engine
Generates:
1. Executive Summary (3-5 simple sentences)
2. Key Takeaways (Bullet points, maximum 5 points)
3. TL;DR (1 sentence)

Uses Gemini / OpenAI AI summarization if configured;
otherwise falls back to high-quality deterministic extractive summarization.

Critical Rule: The summary must explain what the article is about,
NOT whether it is true or false.
"""
import os
import re
import json
import logging
from typing import Dict, Any, List, Optional
import requests

logger = logging.getLogger("lucid.summarizer")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [Summarizer] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
    "by", "can", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for",
    "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just",
    "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once",
    "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "s", "same", "she",
    "should", "so", "some", "such", "t", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "will", "with", "would", "you", "your", "yours", "yourself",
    "yourselves"
}

BOILERPLATE_PATTERNS = [
    r'cookie\s+policy',
    r'all\s+rights\s+reserved',
    r'sign\s+up\s+for\s+(?:our\s+)?newsletter',
    r'subscribe\s+to',
    r'click\s+here\s+to',
    r'read\s+more\s*:',
    r'advertisement',
    r'follow\s+us\s+on',
    r'share\s+this\s+article',
    r'photo\s+credit',
    r'image\s+caption',
    r'terms\s+and\s+conditions',
    r'privacy\s+notice',
    r'published\s+\d+\s+(?:hours?|days?|minutes?)\s+ago',
]


class ArticleSummarizer:
    """
    Summarizes article content objectively into Executive Summary, Key Takeaways, and TL;DR.
    """

    def __init__(self):
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")

    def summarize(self, text: str, title: str = "", url: str = "") -> Dict[str, Any]:
        """
        Main entry point for article summarization.
        Returns:
          {
            "title": str,
            "executiveSummary": str,  # 3-5 simple sentences
            "keyTakeaways": List[str],  # Maximum 5 bullet points
            "tldr": str,  # Exactly 1 sentence
            "provider": str
          }
        """
        if not text or not text.strip():
            return {
                "title": title or "Untitled Article",
                "executiveSummary": "No article content available to summarize.",
                "keyTakeaways": ["No article content provided for analysis."],
                "tldr": "No article text provided.",
                "quickSummary": "No article text provided to summarize.",
                "provider": "Default"
            }

        cleaned_text = self._clean_input_text(text)
        derived_title = title.strip() or self._derive_title(cleaned_text)

        # 1. Try Gemini AI if available
        if self.gemini_key:
            ai_summary = self._query_gemini(cleaned_text, derived_title)
            if ai_summary and self._validate_summary_structure(ai_summary):
                ai_summary["title"] = derived_title
                ai_summary["provider"] = "Gemini AI"
                return ai_summary

        # 2. Try OpenAI if available
        if self.openai_key:
            ai_summary = self._query_openai(cleaned_text, derived_title)
            if ai_summary and self._validate_summary_structure(ai_summary):
                ai_summary["title"] = derived_title
                ai_summary["provider"] = "OpenAI API"
                return ai_summary

        # 3. Fallback to Extractive Summarization
        extractive = self._generate_extractive_summary(cleaned_text, derived_title)
        extractive["title"] = derived_title
        extractive["provider"] = "Extractive NLP Engine"
        return extractive

    def _clean_input_text(self, text: str) -> str:
        """Removes excessive whitespace and standard web artifacts."""
        lines = []
        for line in text.splitlines():
            line_str = line.strip()
            if not line_str:
                continue
            # Skip common web boilerplate lines
            if any(re.search(pat, line_str, re.IGNORECASE) for pat in BOILERPLATE_PATTERNS):
                continue
            lines.append(line_str)
        return "\n".join(lines)

    def _derive_title(self, text: str) -> str:
        """Extracts article headline from first segmented unit/sentence, stripping time and broadcast tags."""
        if not text or not text.strip():
            return "Article Analysis"

        # Check if sentences can be segmented first
        sentences = self._split_sentences(text)
        if sentences:
            for s in sentences:
                candidate = s.strip()
                # Clean time tags like '13m ago: ', '58m ago: '
                candidate = re.sub(r'^(?:\d+\s*[mhd]\s+ago|breaking|watch|live|alert|update)\s*[:—–-]\s*', '', candidate, flags=re.IGNORECASE).strip()
                # Clean broadcast datelines
                candidate = re.sub(r'^[A-Z\s,]{2,20}\s*(?:\([^\)]+\))?\s*[-—–]\s*', '', candidate).strip()
                # Strip trailing punctuation
                if candidate.endswith(('.', '!', '?', ':')):
                    candidate = candidate[:-1].strip()
                if len(candidate) >= 12 and not candidate.startswith(("http://", "https://")):
                    return candidate[:140]

        # Fallback to first paragraph block joined
        first_block = text.strip().split('\n\n')[0].strip()
        flat_block = re.sub(r'\s*\n\s*', ' ', first_block).strip()
        flat_block = re.sub(r'^(?:\d+\s*[mhd]\s+ago|breaking|watch|live|alert|update)\s*[:—–-]\s*', '', flat_block, flags=re.IGNORECASE).strip()
        if flat_block.endswith(('.', '!', '?', ':')):
            flat_block = flat_block[:-1].strip()
        if len(flat_block) >= 10 and not flat_block.startswith(("http://", "https://")):
            return flat_block[:140]

        return "Article Analysis"

    # =================================================================
    # Extractive Summarization Algorithm
    # =================================================================

    def _generate_extractive_summary(self, text: str, title: str) -> Dict[str, Any]:
        """
        Produces extractive Executive Summary (3-5 sentences), Key Takeaways (max 5),
        and TL;DR (1 sentence) without evaluating factual accuracy.
        """
        sentences = self._split_sentences(text)
        if not sentences:
            return {
                "executiveSummary": text[:250].strip() + ("." if not text.endswith(".") else ""),
                "keyTakeaways": [text[:120].strip()],
                "tldr": text[:120].strip()
            }

        title_words = set(re.findall(r'\b[A-Za-z0-9]{3,}\b', title.lower())) - STOP_WORDS

        # Frequency table of significant words in article
        word_freq: Dict[str, int] = {}
        for s in sentences:
            for w in re.findall(r'\b[A-Za-z0-9]{3,}\b', s.lower()):
                if w not in STOP_WORDS:
                    word_freq[w] = word_freq.get(w, 0) + 1

        # Score sentences
        scored_sentences: List[Dict[str, Any]] = []
        for idx, sentence in enumerate(sentences):
            words = re.findall(r'\b[A-Za-z0-9]{3,}\b', sentence.lower())
            word_count = len(sentence.split())

            # Word frequency score
            freq_score = sum(word_freq.get(w, 0) for w in words if w not in STOP_WORDS)
            norm_freq = freq_score / max(1, len(words))

            # Title overlap bonus
            overlap_count = sum(1 for w in words if w in title_words)
            title_bonus = overlap_count * 2.5

            # Position weight (inverted pyramid: earlier sentences carry core facts)
            if idx == 0:
                pos_weight = 3.2
            elif idx == 1:
                pos_weight = 2.6
            elif idx == 2:
                pos_weight = 2.2
            elif idx < 5:
                pos_weight = 1.6
            elif idx < 8:
                pos_weight = 1.2
            else:
                pos_weight = 1.0

            # Concrete fact / entity / number bonus
            concrete_bonus = 0.0
            if re.search(r'\b\d+(?:\.\d+)?%?|\$\d+|\b(?:million|billion|trillion)\b', sentence, re.IGNORECASE):
                concrete_bonus += 1.5
            if re.search(r'\b(?:announced|reported|stated|said|confirmed|launched|approved|declined|increased|rose)\b', sentence, re.IGNORECASE):
                concrete_bonus += 1.2

            # Length normalization (penalize extremely short fragments or overly long run-ons)
            len_mult = 1.0
            if word_count < 8:
                len_mult = 0.4
            elif word_count > 45:
                len_mult = 0.75
            elif 12 <= word_count <= 32:
                len_mult = 1.2

            total_score = (norm_freq + title_bonus + concrete_bonus) * pos_weight * len_mult

            scored_sentences.append({
                "index": idx,
                "text": sentence,
                "score": total_score,
                "word_count": word_count
            })

        # -------------------------------------------------------------
        # 1. TL;DR (Exactly 1 sentence)
        # -------------------------------------------------------------
        # Pick the most representative sentence (often sentence 0 if strong, or highest composite score)
        best_lead = scored_sentences[0]
        highest_scored = max(scored_sentences, key=lambda s: s["score"])
        tldr_sentence = best_lead["text"] if best_lead["score"] >= highest_scored["score"] * 0.75 else highest_scored["text"]
        tldr = self._clean_sentence(tldr_sentence)
        if not tldr.endswith((".", "!", "?")):
            tldr += "."

        # -------------------------------------------------------------
        # 2. Executive Summary (3-5 simple sentences)
        # -------------------------------------------------------------
        # Select between 3 and 5 top sentences, ordered chronologically
        target_count = min(5, max(3, len(scored_sentences)))
        top_by_score = sorted(scored_sentences, key=lambda s: s["score"], reverse=True)[:target_count]
        # Sort chronologically by original text index
        top_chronological = sorted(top_by_score, key=lambda s: s["index"])

        summary_sentences = [self._clean_sentence(s["text"]) for s in top_chronological]
        # Ensure 3-5 sentences
        exec_summary = " ".join(summary_sentences).strip()

        # -------------------------------------------------------------
        # 3. Key Takeaways (Bullet points, maximum 5 points)
        # -------------------------------------------------------------
        key_takeaways: List[str] = []
        seen_stems: List[set] = []

        # Iterate down ranked sentences to find distinct factual takeaways
        ranked_for_bullets = sorted(scored_sentences, key=lambda s: s["score"], reverse=True)
        for cand in ranked_for_bullets:
            cand_text = self._clean_bullet(cand["text"])
            cand_words = set(re.findall(r'\b[A-Za-z0-9]{3,}\b', cand_text.lower())) - STOP_WORDS
            if not cand_words:
                continue

            # Check overlap with already selected takeaways (Jaccard similarity > 0.45)
            is_duplicate = False
            for prev_words in seen_stems:
                jaccard = len(cand_words & prev_words) / max(1, len(cand_words | prev_words))
                if jaccard > 0.45:
                    is_duplicate = True
                    break

            if not is_duplicate:
                key_takeaways.append(cand_text)
                seen_stems.append(cand_words)
                if len(key_takeaways) == 5:
                    break

        if not key_takeaways:
            key_takeaways = [tldr]

        # Quick Summary (1-2 simple sentences in plain English for normal users)
        quick_summary = tldr
        if len(scored_sentences) >= 2:
            sec_sentence = self._clean_sentence(scored_sentences[1]["text"])
            if 6 <= len(sec_sentence.split()) <= 30 and sec_sentence.lower() != tldr.lower():
                quick_summary = f"{tldr} {sec_sentence}"

        return {
            "executiveSummary": exec_summary,
            "keyTakeaways": key_takeaways[:5],
            "tldr": tldr,
            "quickSummary": quick_summary
        }

    def _split_sentences(self, text: str) -> List[str]:
        """Splits article text or OCR headline feeds into clean, individual sentences."""
        if not text or not text.strip():
            return []

        # 1. Break into paragraph or headline blocks
        blocks = [b.strip() for b in re.split(r'\n\s*\n', text.strip()) if b.strip()]
        valid: List[str] = []

        for block in blocks:
            # Check if block contains multiple single-line headlines (e.g. news feeds or bulleted items)
            sub_lines = [l.strip() for l in block.splitlines() if l.strip()]
            current_unit: List[str] = []

            for line in sub_lines:
                # If line starts with a new headline marker (timestamp, bullet, or tag)
                is_headline_start = bool(
                    re.match(r'^(?:\d+\s*[mhd]\s+ago|breaking|watch|live|alert|update|exclusive)\s*[:—–-]', line, re.IGNORECASE)
                    or re.match(r'^(?:[-*•–—]|\d+[\.\)])\s+', line)
                )
                if is_headline_start and current_unit:
                    unit_text = " ".join(current_unit).strip()
                    if unit_text:
                        self._process_and_append_sentence(unit_text, valid)
                    current_unit = [line]
                else:
                    current_unit.append(line)

            if current_unit:
                unit_text = " ".join(current_unit).strip()
                if unit_text:
                    self._process_and_append_sentence(unit_text, valid)

        # Fallback if block splitting yielded fewer than 2 sentences and text has standard punctuation
        if len(valid) < 2:
            flat = re.sub(r'\s*\n\s*', ' ', text.strip())
            raw_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'“‘])', flat)
            if len(raw_sentences) > len(valid):
                valid = []
                for s in raw_sentences:
                    s_clean = s.strip()
                    if len(s_clean.split()) >= 4 and re.search(r'[a-zA-Z]', s_clean):
                        valid.append(s_clean)

        return valid

    def _process_and_append_sentence(self, raw_text: str, target_list: List[str]):
        """Cleans timestamps and splits internal sentence punctuation if any."""
        # Strip leading time markers like '13m ago: ', '28m ago: '
        cleaned = re.sub(r'^(?:\d+\s*[mhd]\s+ago|breaking|watch|live)\s*[:—–-]\s*', '', raw_text, flags=re.IGNORECASE).strip()
        # Split on standard punctuation within the headline if present
        sub_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'“‘])', cleaned)
        for s in sub_sentences:
            s_clean = s.strip()
            if len(s_clean.split()) >= 4 and re.search(r'[a-zA-Z]', s_clean):
                if not s_clean.endswith(('.', '!', '?')):
                    s_clean += '.'
                target_list.append(s_clean)

    def _clean_sentence(self, sentence: str) -> str:
        """Cleans formatting, leading discourse markers, and punctuation."""
        s = sentence.strip()
        # Remove leading list bullets or numbers
        s = re.sub(r'^(?:[-*•–—]|\d+[\.\)])\s*', '', s)
        # Remove leading broadcast/wire tags: e.g. "WATCH: ", "BREAKING: ", "ALERT: "
        s = re.sub(r'^(?:WATCH|BREAKING|LIVE|ALERT|UPDATE)\s*[:—–-]\s*', '', s, flags=re.IGNORECASE)
        # Remove leading journalistic datelines: e.g. "LONDON (Reuters) — "
        s = re.sub(r'^[A-Z\s,]{2,20}\s*(?:\([^\)]+\))?\s*[-—–]\s*', '', s)
        s = s.strip()
        if not s.endswith((".", "!", "?")):
            s += "."
        return s

    def _clean_bullet(self, sentence: str) -> str:
        """Formats a sentence into an active, clear bullet point."""
        s = self._clean_sentence(sentence)
        # Trim conversational leading transitions
        s = re.sub(r'^(?:However|Moreover|Furthermore|In addition|Meanwhile|Notably|Consequently),\s*', '', s, flags=re.IGNORECASE)
        # Capitalize first letter
        if s:
            s = s[0].upper() + s[1:]
        return s

    def _validate_summary_structure(self, data: Dict[str, Any]) -> bool:
        """Validates that AI output contains executiveSummary, keyTakeaways (<=5), and tldr."""
        if not isinstance(data, dict):
            return False
        if not data.get("executiveSummary") or not isinstance(data["executiveSummary"], str):
            return False
        if not data.get("keyTakeaways") or not isinstance(data["keyTakeaways"], list):
            return False
        if not data.get("tldr") or not isinstance(data["tldr"], str):
            return False
        # Normalize takeaways to max 5
        data["keyTakeaways"] = [str(x).strip() for x in data["keyTakeaways"][:5] if str(x).strip()]
        if not data.get("keyTakeaways"):
            return False
        if not data.get("quickSummary"):
            data["quickSummary"] = data.get("tldr", "")
        return True

    # =================================================================
    # External LLM Integrations (Gemini / OpenAI)
    # =================================================================

    def _query_gemini(self, text: str, title: str) -> Optional[Dict[str, Any]]:
        """Queries Gemini for objective article summarization."""
        try:
            model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            prompt = (
                "You are an objective article summarizer for the Lucid news verification platform.\n"
                "Your task is to explain what this article is about in neutral, clear, and informative prose.\n"
                "CRITICAL REQUIREMENT: Do NOT judge or comment on whether the claims are true, false, verified, or debunked.\n"
                "Focus strictly on summarizing what the article discusses and reports.\n\n"
                f"Article Title: {title}\n"
                f"Article Text:\n{text[:4000]}\n\n"
                "Return ONLY a valid JSON object matching this exact structure:\n"
                "{\n"
                '  "executiveSummary": "A coherent executive summary of 3 to 5 simple sentences explaining what the article is about.",\n'
                '  "keyTakeaways": [\n'
                '    "Bullet point 1",\n'
                '    "Bullet point 2",\n'
                '    "Bullet point 3",\n'
                '    "Bullet point 4",\n'
                '    "Bullet point 5"\n'
                "  ],\n"
                '  "tldr": "Exactly 1 concise sentence summarizing the main event or topic."\n'
                "}\n"
                "Ensure 'keyTakeaways' contains a maximum of 5 bullet points.\n"
                "Ensure 'executiveSummary' contains between 3 and 5 simple sentences.\n"
                "Ensure 'tldr' is exactly 1 sentence."
            )
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"response_mime_type": "application/json"}
            }
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": self.gemini_key
            }
            res = requests.post(url, headers=headers, json=payload, timeout=8)
            if res.status_code == 200:
                body = res.json()
                raw_text = body["candidates"][0]["content"]["parts"][0]["text"]
                parsed = json.loads(raw_text)
                return parsed
        except Exception as e:
            logger.warning(f"Gemini summarization fallback: {e}")
        return None

    def _query_openai(self, text: str, title: str) -> Optional[Dict[str, Any]]:
        """Queries OpenAI for objective article summarization."""
        try:
            headers = {
                "Authorization": f"Bearer {self.openai_key}",
                "Content-Type": "application/json"
            }
            prompt = (
                "You are an objective article summarizer for the Lucid news verification platform.\n"
                "Explain what this article is about in neutral, factual prose.\n"
                "CRITICAL: Do NOT evaluate or state whether the claims are true, false, verified, or debunked.\n"
                f"Article Title: {title}\n\n"
                f"Article Text:\n{text[:4000]}\n\n"
                "Return ONLY a valid JSON object with keys:\n"
                "- 'executiveSummary': 3 to 5 simple sentences explaining what the article is about.\n"
                "- 'keyTakeaways': Array of bullet points (maximum 5 points).\n"
                "- 'tldr': Exactly 1 sentence summarizing the core topic."
            )
            payload = {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"}
            }
            res = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=8)
            if res.status_code == 200:
                body = res.json()
                raw = body["choices"][0]["message"]["content"]
                parsed = json.loads(raw)
                return parsed
        except Exception as e:
            logger.warning(f"OpenAI summarization fallback: {e}")
        return None


summarizer = ArticleSummarizer()

def get_article_summarizer() -> ArticleSummarizer:
    return summarizer
