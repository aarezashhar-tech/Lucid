"""
Layer 2: Claim Extraction Engine with Content-Cleaning Stage
Decomposes articles and text submissions into distinct, testable factual propositions.
Filters out navigation elements, author names, image captions, and UI artifacts.
"""
import re
from typing import List, Dict, Any

# =====================================================================
# Content-Cleaning Patterns: UI artifacts, Navigation, Bylines, Captions
# =====================================================================

UI_NOISE_PATTERNS = [
    # UI widgets & Newsroom widgets
    r'add\s+as\s+preferred(\s+on\s+google)?',
    r'follow\s+(us\s+)?(on\s+)?(google\s+news|twitter|x|facebook|instagram|whatsapp|telegram|threads)',
    r'download\s+(the\s+)?(app|application|dossier)',
    r'subscribe(\s+now)?(\s+for\s+free)?',
    r'sign\s+up(\s+for\s+our\s+newsletter)?',
    r'join\s+our\s+channel',
    r'listen\s+to\s+this\s+(article|story|audio)',
    r'share\s+(this\s+)?(story|article|page|post)?',
    r'^(share|save|bookmark|copy\s+link|print|email)\b',

    # Navigation & Related links
    r'^(related|related\s+topics?|related\s+stories?|related\s+articles?|more\s+on\s+this\s+story|see\s+also|read\s+more|read\s+also)\b',
    r'^(top\s+stories|most\s+read|latest\s+updates|popular\s+stories|must\s+reads?|trending)\b',
    r'^(home|news|world|middle\s+east|politics|business|technology|tech|science|health|sport|entertainment|opinion|analysis)\b',

    # Category & Topic tags
    r'^(topics?:?|categories?:?|tags?:?|filed\s+under:?)\b',

    # Image / Media captions & credits
    r'^(image\s+caption|photo\s+caption|image\s+source|photo\s+credit|photo:?|image:?|credit:?|getty\s+images|reuters/|ap\s+photo|afp/|epa-efe|file\s+photo|courtesy\s+of)\b',

    # Advertisements & Legal boilerplate
    r'^(advertisement|sponsored\s+content|promoted\s+stories|ad\s+content|cookie\s+policy|terms\s+of\s+(use|service)|all\s+rights\s+reserved|copyright\s+\d{4})\b',

    # Byline and author prefixes
    r'^(by|reported\s+by|written\s+by|edited\s+by|words\s+by|author:?|correspondent:?)\b',
]

# Common action, state, and reporting verbs (finite, participles, and common stems)
VERB_PATTERN = re.compile(
    r'\b('
    r'is|are|was|were|be|been|being|has|have|had|do|does|did|will|would|shall|should|can|could|may|might|must|'
    r'discuss|discusses|discussed|discussing|return|returns|returned|returning|come|comes|came|coming|back|backs|backed|backing|'
    r'step|steps|stepped|stepping|resign|resigns|resigned|resigning|replace|replaces|replaced|replacing|'
    r'appoint|appoints|appointed|appointing|consider|considers|considered|considering|plan|plans|planned|planning|'
    r'meet|meets|met|meeting|seek|seeks|sought|seeking|urge|urges|urged|urging|call|calls|called|calling|'
    r'vow|vows|vowed|vowing|pledge|pledges|pledged|pledging|weigh|weighs|weighed|weighing|'
    r'endorse|endorses|endorsed|endorsing|endorsement|back|backs|backed|backing|support|supports|supported|supporting|'
    r'reject|rejects|rejected|rejecting|denounce|denounces|denounced|condemn|condemns|condemned|'
    r'hire|hires|hired|hiring|fire|fires|fired|firing|cut|cuts|cutting|win|wins|won|winning|lose|loses|lost|losing|'
    r'shot|killed|died|injured|wounded|hit|struck|destroyed|damaged|intercepted|launched|bombed|shelled|attacked|'
    r'signed|approved|passed|voted|ratified|adopted|agreed|decided|enacted|ordered|authorized|mandated|'
    r'reported|confirmed|denied|stated|announced|claimed|declared|warned|testified|admitted|rebutted|reiterated|'
    r'found|discovered|revealed|uncovered|showed|demonstrated|proved|concluded|indicated|suggested|observed|'
    r'rose|increased|grew|fell|dropped|declined|surpassed|exceeded|doubled|tripled|reached|totaled|measured|'
    r'evacuated|rescued|arrested|detained|charged|sentenced|convicted|sued|released|freed|fled|displaced|'
    r'said|told|spoke|commented|wrote|posted|tweeted|broadcast|published|noted|argued|'
    r'began|started|ended|concluded|halted|stopped|delayed|continued|occurred|happened|sparked|'
    r'continue|continues|continuing|serve|serves|served|serving|remain|remains|remained|remaining|'
    r'hold|holds|held|holding|act|acts|acted|acting|become|becomes|became|becoming|lead|leads|led|leading|'
    r'cause|causes|caused|causing|control|controls|controlled|controlling|'
    r'contain|contains|contained|containing|affect|affects|affected|affecting|'
    r'alter|alters|altered|infect|infects|infected|transmit|transmits|transmitted|'
    r'spread|spreads|spreading|leak|leaks|leaked|hide|hides|hidden|implant|implants|implanted|'
    r'drink|drinks|drinking|drank|drunk|produce|produces|produced|create|creates|created|'
    r'lead|leads|led|leading|cure|cures|cured|prevent|prevents|prevented|harm|harms|harmed|'
    r'\w+ed|\w+ing'  # Regular past tense and participles
    r')\b',
    re.IGNORECASE
)

# Multilingual action, state, and reporting verb patterns (Spanish, French, German, Portuguese, Italian)
MULTILINGUAL_VERB_PATTERN = re.compile(
    r'\b('
    # Spanish & Portuguese
    r'es|son|fue|fueron|era|eran|ha|han|hay|tiene|tienen|tenía|dice|dicen|dijo|dijeron|'
    r'afirma|afirman|afirmó|anuncia|anuncian|anunció|declara|declaran|declaró|'
    r'está|están|estaba|estaban|ocurrió|pasó|informa|informó|reporta|reportaron|'
    r'foi|foram|tem|têm|disse|disseram|afirmou|anunciou|declarou|estava|estavam|'
    # French
    r'est|sont|était|étaient|a|ont|avait|avaient|dit|disent|déclaré|déclarent|'
    r'affirme|affirment|annonce|annoncent|annoncé|semble|semblent|trouve|trouvent|'
    # German
    r'ist|sind|war|waren|hat|haben|hatte|hatten|wird|werden|wurde|wurden|'
    r'sagt|sagten|berichtet|berichten|meldet|melden|erklärt|erklärten|bestätigt|'
    # Italian
    r'è|sono|era|erano|ha|hanno|aveva|avevano|dice|dicono|detto|dichiarato|dichiara|'
    r'annuncia|annunciato|riporta|riportano|'
    # Russian Cyrillic common reporting stems (Latin translit or Cyrillic)
    r'заявил|сообщил|сказал|отметил|передает|утверждает|является|было|были'
    r')\b',
    re.IGNORECASE
)

# Non-Latin script detector
NON_LATIN_SCRIPT_PATTERN = re.compile(
    r'[\u0600-\u06FF\u0400-\u04FF\u0900-\u097F\u4E00-\u9FFF\u3040-\u30FF\uAC00-\uD7AF\u0590-\u05FF]'
)

# Subjective and opinion statement indicators (Requirement 9: Do not fact-check opinion paragraphs)
OPINION_SENTENCE_REGEX = re.compile(
    r'\b('
    r'in my opinion|in our view|in my view|i believe|we believe|i think|we think|i feel|we feel|'
    r'from my perspective|in my judgment|it seems to me|should be|must be|ought to be|'
    r'unacceptable|outrageous|disastrous|shameful|foolish|brilliant|inspiring|unjust|immoral'
    r')\b',
    re.IGNORECASE
)
KNOWN_ACRONYMS = {
    'CEO', 'CTO', 'CFO', 'COO', 'NVIDIA', 'AI', 'US', 'USA', 'UK', 'EU', 'UN', 'WHO', 'IEA',
    'BBC', 'CNN', 'NASA', 'FBI', 'CIA', 'NATO', 'GOP', 'IDF', 'PM', 'VP', 'SEC', 'FTC', 'IMF',
    'GDP', 'IPO', 'EV', 'COVID', 'FDA', 'CDC', 'AMD', 'LLM', 'GPU', 'CPU'
}

LOWERCASE_WORDS = {
    'a', 'an', 'the', 'as', 'at', 'by', 'for', 'in', 'of', 'on', 'per', 'to', 'up', 'with',
    'and', 'but', 'or', 'nor', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'has', 'have',
    'had', 'do', 'does', 'did', 'coming', 'back', 'discussing', 'discusses', 'return', 'returns'
}

COMMON_HELPERS = {
    'in', 'at', 'on', 'as', 'by', 'for', 'to', 'with', 'from', 'is', 'are', 'was', 'were',
    'has', 'have', 'had', 'and', 'or', 'of', 'over', 'into', 'after', 'before'
}


def normalize_claim_text(text: str) -> str:
    """
    Normalizes OCR / headline-style text into clean grammatical propositions.
    Example:
      'JENSEN HUANG DISCUSSES COMING BACK AS NVIDIA CEO'
      -> 'Jensen Huang is discussing a return as NVIDIA CEO.'
    """
    if not text:
        return ""

    s = text.strip()

    # Check if original input was predominantly uppercase (OCR headlines)
    letters = [c for c in s if c.isalpha()]
    is_headline_caps = len(letters) > 0 and (sum(1 for c in letters if c.isupper()) / len(letters)) > 0.6

    # Normalize headline phrases to proper claim grammar
    s = re.sub(r'\bdiscuss(?:es|ed)?\s+coming\s+back\s+as\b', 'is discussing a return as', s, flags=re.IGNORECASE)
    s = re.sub(r'\bdiscuss(?:es|ed)?\s+returning\s+as\b', 'is discussing a return as', s, flags=re.IGNORECASE)
    s = re.sub(r'\bdiscuss(?:es|ed)?\s+stepping\s+down\s+as\b', 'is discussing stepping down as', s, flags=re.IGNORECASE)
    s = re.sub(r'\bweighs?\s+(?:a\s+)?return\s+as\b', 'is weighing a return as', s, flags=re.IGNORECASE)
    s = re.sub(r'\bconsiders?\s+coming\s+back\s+as\b', 'is considering a return as', s, flags=re.IGNORECASE)
    s = re.sub(r'\bplans?\s+(?:to\s+return|a\s+return)\s+as\b', 'is planning a return as', s, flags=re.IGNORECASE)

    words = s.split()
    norm_words = []
    for i, w in enumerate(words):
        m = re.match(r'^([^\w]*)(.*?)([^\w]*)$', w)
        if m:
            prefix, core, suffix = m.groups()
        else:
            prefix, core, suffix = '', w, ''
        u = core.upper()
        l = core.lower()

        if u in KNOWN_ACRONYMS:
            token = u
        elif i > 0 and l in LOWERCASE_WORDS:
            token = l
        elif is_headline_caps and core.isupper():
            token = core.capitalize()
        else:
            token = core
        norm_words.append(prefix + token + suffix)

    res = ' '.join(norm_words).strip()
    if not res.endswith(('.', '!', '?')):
        res += '.'
    return res

# Author name standalone pattern (e.g. "David Gritten", "David Gritten, BBC News", "David Gritten in Jerusalem")
# Case-sensitive to avoid matching uppercase news verbs like "DISCUSSES COMING"
AUTHOR_NAME_PATTERN = re.compile(
    r'^[A-Z][a-z]{1,20}(\s+[A-Z][a-z]{1,20}){1,2}(\s*,\s*(BBC\s+News|Reuters|AP|AFP|CNN|Al\s+Jazeera|correspondent|editor|reporter|in\s+[A-Z][a-z]+))?$'
)

# Isolated category / location words
ISOLATED_TOPIC_WORDS = {
    "israel", "west bank", "gaza", "middle east", "jerusalem", "washington", "london", "geneva",
    "paris", "beijing", "moscow", "kyiv", "ukraine", "russia", "related", "world", "news",
    "politics", "business", "technology", "tech", "science", "health", "sport", "entertainment"
}

TRUISM_ABSTRACT_SUBJECTS = {
    "hope", "life", "love", "peace", "kindness", "honesty", "success", "happiness",
    "truth", "time", "courage", "music", "art", "beauty", "friendship", "wisdom",
    "patience", "change", "faith", "freedom", "justice", "fear", "knowledge",
    "silence", "forgiveness", "destiny", "dream", "dreams"
}

TRUISM_PHRASES = [
    r'\bmore\s+than\s+(?:a\s+|just\s+a\s+)?game\b',
    r'\bjourney\s+of\s+self\b',
    r'\bbest\s+policy\b',
    r'\bbegins?\s+with\s+a\s+smile\b',
    r'\bconquers?\s+all\b',
    r'\bonly\s+constant\b',
    r'\bthing\s+with\s+feathers\b',
    r'\bkey\s+to\s+success\b',
    r'\bsilver\s+lining\b',
    r'\bmeaning\s+of\s+life\b',
    r'\bwhere\s+there\s+is\s+(?:hope|love|life)\b',
    r'\bchange\s+the\s+world\b',
    r'\bmatter\s+of\s+time\b',
    r'\bpower\s+of\s+(?:love|hope|kindness|dreams?)\b',
    r'\bcost(?:s)?\s+nothing\b',
]


def is_generic_philosophical_truism(sentence: str) -> bool:
    """
    Identifies pure philosophical/generic truisms that contain no verifiable
    person, organization, place, date, or event.
    Examples:
      'Football is more than just a game.'
      'Hope is what keeps humanity alive.'
      'Life is a journey of self discovery.'
    """
    if not sentence:
        return False

    s_clean = sentence.strip().strip('"\'“”‘’').strip()
    words = s_clean.split()
    if len(words) < 2:
        return False

    s_lower = s_clean.lower()

    # 1. Do not skip if the sentence contains verifiable concrete markers:
    # Numbers, dates, years
    if re.search(r'\b(?:\d{1,4}|19\d\d|20\d\d)\b', s_clean):
        return False

    # Geographic locations
    COMMON_GEO_PATTERNS = (
        r'\b(al-mawasi|mawasi|gaza|palestine|israel|west\s+bank|jerusalem|'
        r'ukraine|russia|kyiv|moscow|us|usa|uk|britain|france|paris|germany|'
        r'berlin|china|beijing|india|delhi|egypt|cairo|lebanon|beirut|syria|'
        r'damascus|iran|tehran|iraq|baghdad|yemen|sanaa|qatar|doha|uae|dubai|earth|moon)\b'
    )
    if re.search(COMMON_GEO_PATTERNS, s_lower):
        return False

    # Specific roles, institutions, or public offices
    ROLE_AND_INSTITUTION_PATTERNS = (
        r'\b(coach|president|prime\s+minister|minister|ceo|cfo|judge|court|'
        r'parliament|senate|congress|police|military|soldier|army|navy|'
        r'hospital|doctor|university|school|company|corporation|bank|treaty|'
        r'tournament|cup|olympics|fifa|uefa|un|who|fda|cdc|nasa|fbi|cia)\b'
    )
    if re.search(ROLE_AND_INSTITUTION_PATTERNS, s_lower):
        return False

    # Capitalized named entities (excluding sentence starters)
    capitalized_entities = [
        w.strip('.,!?:;"\'()') for w in words[1:]
        if w and w[0].isupper() and len(w) > 2 and w.upper() not in KNOWN_ACRONYMS
    ]
    proper_entities = [
        w for w in capitalized_entities
        if w.lower() not in LOWERCASE_WORDS and w.lower() not in TRUISM_ABSTRACT_SUBJECTS
    ]
    if len(proper_entities) >= 1:
        return False

    # Known acronyms (e.g. NATO, EU, UN, NASA, COVID, 5G)
    if any(acr in s_clean for acr in KNOWN_ACRONYMS):
        return False

    # 2. Check if it matches truism subject or phrase:
    first_w = words[0].lower().strip('.,!?:;"\'()')
    if first_w in TRUISM_ABSTRACT_SUBJECTS:
        return True

    if any(re.search(pat, s_lower) for pat in TRUISM_PHRASES):
        return True

    if re.match(r'^(?:football|soccer|art|music|love|hope|life|kindness|honesty|peace)\s+(?:is|can|will|matters|brings|comes|requires)\b', s_lower):
        return True

    return False


def extract_speaker_attribution(text: str) -> Optional[str]:
    """
    Detects speaker name from quote-cards, interviews, bylines, or attribution markers.
    Returns the clean speaker name (e.g. 'Ehab Abu Jazar').
    """
    if not text:
        return None

    # 1. Dash / Hyphen attribution: "- Ehab Abu Jazar, Palestine football coach" or "— Ehab Abu Jazar"
    dash_match = re.search(r'[-–—~]\s*([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+)(?:\s*,\s*[^.\n]+)?', text)
    if dash_match:
        cand = dash_match.group(1).strip()
        if cand.lower() not in ISOLATED_TOPIC_WORDS and len(cand.split()) >= 2:
            return cand

    # 2. Colon attribution: "Ehab Abu Jazar, Palestine football coach: ..." or "Ehab Abu Jazar: ..."
    colon_match = re.search(r'(?:^|\n)\s*([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+)(?:\s*,\s*[^:\n]+)?\s*:\s*["“\'‘]?', text)
    if colon_match:
        cand = colon_match.group(1).strip()
        if cand.lower() not in ISOLATED_TOPIC_WORDS and len(cand.split()) >= 2:
            return cand

    # 3. Reporting verb attribution: "..., said Ehab Abu Jazar" or "according to Ehab Abu Jazar"
    rep_match = re.search(
        r'\b(?:said|stated|told|says|warns?|explains?|noted|according\s+to)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+)\b',
        text
    )
    if rep_match:
        cand = rep_match.group(1).strip()
        if cand.lower() not in ISOLATED_TOPIC_WORDS and len(cand.split()) >= 2:
            return cand

    # 4. Speaker before reporting verb: "Ehab Abu Jazar stated: ..." or "Ehab Abu Jazar said"
    pre_rep_match = re.search(
        r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+)\s+(?:said|stated|told|says|warned|explained|noted)\b',
        text
    )
    if pre_rep_match:
        cand = pre_rep_match.group(1).strip()
        if cand.lower() not in ISOLATED_TOPIC_WORDS and len(cand.split()) >= 2:
            return cand

    # 5. Reverse line inspection for trailing byline / speaker line in quote cards
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for line in reversed(lines):
        clean_l = re.sub(r'^[-–—~•*]\s*', '', line).strip()
        m = re.match(r'^([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+)(?:\s*,\s*.*)?$', clean_l)
        if m:
            cand = m.group(1).strip()
            if not VERB_PATTERN.search(clean_l) and cand.lower() not in ISOLATED_TOPIC_WORDS and len(cand.split()) >= 2:
                return cand

    return None


def rewrite_first_person_quote(sentence: str, speaker: str) -> str:
    """
    Rewrites first-person pronouns in a quoted sentence to third-person
    and attributes the claim to the named speaker.
    Example:
      'My family is living in tents in Al-Mawasi' -> 'Ehab Abu Jazar stated his family is living in tents in Al-Mawasi.'
    """
    if not sentence or not speaker:
        return sentence

    s = sentence.strip().strip('"\'“”‘’').strip()
    # Strip any trailing attribution tag if attached: " - Ehab Abu Jazar..."
    s = re.sub(r'\s*[-–—~]\s*' + re.escape(speaker) + r'(?:\s*,\s*[^.\n]+)?\.?$', '', s, flags=re.I).strip()
    s = re.sub(r'\s*[-–—~]\s*[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+(?:\s*,\s*[^.\n]+)?\.?$', '', s).strip()

    # Clean up trailing ellipsis or punctuation
    s = re.sub(r'\.{2,}$', '', s).strip()
    if not s:
        return sentence

    # If speaker is already mentioned in the sentence, no need to prepend attribution
    if speaker.lower() in s.lower():
        return s if s.endswith(('.', '!', '?')) else s + '.'

    # Check for first-person pronouns
    has_first_person = bool(re.search(r'\b(i|me|my|mine|myself|we|us|our|ours|ourselves)\b', s, re.IGNORECASE))

    if not has_first_person:
        # If no first-person pronouns, check if this is an explicit quoted statement that needs attribution
        first_word = s.split()[0] if s.split() else ""
        rest = s[len(first_word):]
        if first_word.isupper() and len(first_word) <= 4:
            clean_start = first_word
        elif first_word not in ('I', 'God', 'Allah', 'UN', 'WHO', 'US', 'UK', 'EU', 'IDF'):
            clean_start = first_word[0].lower() + first_word[1:] if len(first_word) > 1 else first_word.lower()
        else:
            clean_start = first_word
        attributed = f"{speaker} stated {clean_start}{rest}"
        if not attributed.endswith(('.', '!', '?')):
            attributed += '.'
        return attributed

    # Systematic pronoun replacements
    # 1. Phrases
    s = re.sub(r'\bI am\b', 'he is', s)
    s = re.sub(r'\bi am\b', 'he is', s)
    s = re.sub(r'\bI\'m\b', 'he is', s)
    s = re.sub(r'\bi\'m\b', 'he is', s)
    s = re.sub(r'\bI was\b', 'he was', s)
    s = re.sub(r'\bI have\b', 'he has', s)
    s = re.sub(r'\bI\'ve\b', 'he has', s)
    s = re.sub(r'\bI will\b', 'he will', s)
    s = re.sub(r'\bI\'ll\b', 'he will', s)
    s = re.sub(r'\bI would\b', 'he would', s)
    s = re.sub(r'\bI\'d\b', 'he would', s)
    s = re.sub(r'\bWe are\b', 'they are', s)
    s = re.sub(r'\bwe are\b', 'they are', s)
    s = re.sub(r'\bWe\'re\b', 'they are', s)
    s = re.sub(r'\bwe\'re\b', 'they are', s)
    s = re.sub(r'\bWe have\b', 'they have', s)
    s = re.sub(r'\bwe have\b', 'they have', s)
    s = re.sub(r'\bWe\'ve\b', 'they have', s)
    s = re.sub(r'\bwe\'ve\b', 'they have', s)
    s = re.sub(r'\bWe were\b', 'they were', s)
    s = re.sub(r'\bwe were\b', 'they were', s)
    s = re.sub(r'\bWe will\b', 'they will', s)
    s = re.sub(r'\bwe will\b', 'they will', s)
    s = re.sub(r'\bWe\'ll\b', 'they will', s)
    s = re.sub(r'\bwe\'ll\b', 'they will', s)

    # 2. Singular tokens
    s = re.sub(r'\bMy\b', 'his', s)
    s = re.sub(r'\bmy\b', 'his', s)
    s = re.sub(r'\bMine\b', 'his', s)
    s = re.sub(r'\bmine\b', 'his', s)
    s = re.sub(r'\bMyself\b', 'himself', s)
    s = re.sub(r'\bmyself\b', 'himself', s)
    s = re.sub(r'\bI\b', 'he', s)
    s = re.sub(r'\bme\b', 'him', s)

    # 3. Plural tokens
    s = re.sub(r'\bOur\b', 'their', s)
    s = re.sub(r'\bour\b', 'their', s)
    s = re.sub(r'\bOurs\b', 'theirs', s)
    s = re.sub(r'\bours\b', 'theirs', s)
    s = re.sub(r'\bOurselves\b', 'themselves', s)
    s = re.sub(r'\bourselves\b', 'themselves', s)
    s = re.sub(r'\bWe\b', 'they', s)
    s = re.sub(r'\bwe\b', 'they', s)
    s = re.sub(r'\bus\b', 'them', s)

    words = s.split()
    if words:
        first_w = words[0]
        if first_w.lower() in ('his', 'he', 'they', 'their', 'the', 'a', 'an', 'that'):
            first_w = first_w.lower()
        rest_s = " ".join([first_w] + words[1:])
    else:
        rest_s = s

    attributed = f"{speaker} stated {rest_s}"
    if not attributed.endswith(('.', '!', '?')):
        attributed += '.'
    return attributed


def split_compound_claim(sentence: str) -> List[str]:
    """
    Requirement 4: Splits compound claims connected by conjunctions ('and', 'while', 'but', etc.)
    into independent, verifiable sub-claims.
    Only splits when both parts constitute independent factual propositions.
    """
    if not sentence or not isinstance(sentence, str):
        return [sentence] if sentence else []

    s_clean = sentence.strip()
    delims = [', and ', ', but ', ', while ', '; ', ' while ', ' whereas ', ' and ', ' but ']
    for d in delims:
        if d in s_clean:
            parts = s_clean.split(d, 1)
            p1 = parts[0].strip()
            p2 = parts[1].strip()
            # Both parts must have at least 3 words to avoid splitting compound nouns or lists
            if len(p1.split()) >= 3 and len(p2.split()) >= 2:
                v1 = bool(VERB_PATTERN.search(p1))
                v2 = bool(VERB_PATTERN.search(p2))
                if v1 and v2:
                    words2 = p2.split()
                    # If p2 starts with a verb or auxiliary, carry over the leading subject of p1
                    if VERB_PATTERN.match(words2[0]) or words2[0].lower() in ('also', 'then'):
                        words1 = p1.split()
                        subj_words = []
                        for w in words1:
                            if VERB_PATTERN.match(w):
                                break
                            subj_words.append(w)
                        if subj_words:
                            subj_prefix = ' '.join(subj_words)
                            p2 = f'{subj_prefix} {p2}'
                    if p2 and p2[0].islower():
                        p2 = p2[0].upper() + p2[1:]
                    return [p1, p2]
    return [s_clean]


class ClaimExtractionEngine:
    """
    Decomposes articles into verified factual propositions.
    Includes an active content-cleaning stage before extraction.
    """

    def clean_article_content(self, text: str) -> str:
        """
        Stage 1: Content-Cleaning Stage
        Removes navigation text, author bylines, UI buttons, image captions,
        OCR artifacts, and advertising boilerplate before claim extraction.
        Preserves paragraph boundaries and joins non-punctuated OCR lines.
        """
        if not text or not text.strip():
            return ""

        # Normalize line breaks and group into paragraphs
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n+', text.strip()) if p.strip()]
        cleaned_paragraphs: List[str] = []

        for p in paragraphs:
            lines = [l.strip() for l in p.splitlines() if l.strip()]
            cleaned_lines: List[str] = []
            for line in lines:
                line_clean = line.strip().strip('"\'“”')

                # Filter out garbled OCR glyph noise (e.g. "/ @¢ ad", "*#@!")
                alnums = [c for c in line_clean if c.isalnum()]
                if len(alnums) <= 2 and len(line_clean) < 10:
                    continue
                if len(alnums) / max(1, len(line_clean)) < 0.4 and len(line_clean) < 15:
                    continue

                # Check if line matches UI noise patterns
                if any(re.search(pat, line_clean, re.IGNORECASE) for pat in UI_NOISE_PATTERNS):
                    continue

                # Check if line is an isolated author name (only if no action verbs are present)
                if not VERB_PATTERN.search(line_clean) and AUTHOR_NAME_PATTERN.match(line_clean):
                    continue

                # Check if line is an isolated topic / category label
                if line_clean.lower().strip('.:;,') in ISOLATED_TOPIC_WORDS:
                    continue

                # Strip leading byline prefix if merged into paragraph: "By David Gritten. Israeli forces..." -> "Israeli forces..."
                line_stripped = re.sub(r'^(by\s+[A-Z][a-z]+(\s+[A-Z][a-z]+){1,2}\s*[\.:,-]\s*)', '', line_clean, flags=re.IGNORECASE)

                # Strip leading datelines if present: "JERUSALEM (Reuters) - Israeli forces..." -> "Israeli forces..."
                line_stripped = re.sub(r'^[A-Z\s,]+(\([A-Za-z\s,]+\))?\s*[-–—]\s*', '', line_stripped)

                # Strip leading broadcast / news alert prefixes: "JUST IN: ...", "BREAKING: ..."
                line_stripped = re.sub(r'^(just\s+in|breaking(\s+news)?|exclusive|alert|update|live|watch|report)\s*[:—–-]\s*', '', line_stripped, flags=re.IGNORECASE)

                if line_stripped:
                    cleaned_lines.append(line_stripped)

            if cleaned_lines:
                # In OCR, multi-line headlines without terminal punctuation are joined
                cleaned_paragraphs.append(" ".join(cleaned_lines))

        return "\n\n".join(cleaned_paragraphs)

    def is_valid_factual_claim(self, sentence: str) -> bool:
        """
        Validates that a candidate statement is an actual verifiable factual proposition,
        not an author name, UI label, category tag, or fragment.
        """
        s = sentence.strip().strip('"').strip("'").strip('“').strip('”').strip('.').strip()
        words = s.split()

        # 1. Word count threshold: factual claims require at least 3 words (e.g. "Earth is flat", "5G causes COVID-19")
        if len(words) < 3:
            return False

        # 1b. Skip pure philosophical/generic truisms that contain no verifiable person, organization, place, date, or event
        if is_generic_philosophical_truism(s):
            return False

        # 2. Blacklist check against UI, navigation, captions, and widgets
        for pat in UI_NOISE_PATTERNS:
            if re.search(pat, s, re.IGNORECASE):
                return False

        # 3. Check for standalone author / reporter names
        if AUTHOR_NAME_PATTERN.match(s):
            return False

        # 4. Check for isolated category, topic, or location labels
        if s.lower().strip('.:;,') in ISOLATED_TOPIC_WORDS:
            return False

        # 5. Filter out questions and imperative call-to-actions
        if s.endswith('?') or re.match(r'^(click|visit|download|follow|subscribe|join|read|check\s+out|share|comment)\b', s, re.IGNORECASE):
            return False

        # 6. Must contain at least one finite action or reporting verb (English, multilingual, or non-Latin script)
        has_verb = (
            bool(VERB_PATTERN.search(s)) or
            bool(MULTILINGUAL_VERB_PATTERN.search(s)) or
            bool(NON_LATIN_SCRIPT_PATTERN.search(s))
        )
        if not has_verb:
            return False

        # 7. Check proper noun fragment (e.g. "David Gritten", "West Bank", "Middle East")
        # Do NOT reject all-caps headlines (e.g. "JENSEN HUANG DISCUSSES COMING BACK AS NVIDIA CEO") or sentences containing verbs
        if all(w and w[0].isupper() for w in words) and not s.isupper():
            if not any(w.lower() in COMMON_HELPERS for w in words):
                return False

        return True

    def split_into_sentences(self, text: str) -> List[str]:
        """Splits cleaned text into grammatical sentences, joining non-punctuated OCR lines."""
        if not text:
            return []

        # In OCR / social graphics, headlines span multiple lines without terminal punctuation
        paragraphs = re.split(r'\n\s*\n+', text.strip())
        merged_paragraphs = []
        for p in paragraphs:
            lines = [l.strip() for l in p.splitlines() if l.strip()]
            joined = ""
            for l in lines:
                if joined:
                    if joined.endswith(('.', '!', '?')):
                        joined += " " + l
                    else:
                        joined += " " + l
                else:
                    joined = l
            if joined:
                merged_paragraphs.append(joined)

        normalized = ". ".join(merged_paragraphs)
        normalized = re.sub(r'[ \t]+', ' ', normalized)

        # Protect honorifics and abbreviations with periods so they do not break sentences mid-entity
        abbrs = [
            'Mr', 'Mrs', 'Ms', 'Dr', 'Prof', 'Gov', 'Sen', 'Rep', 'Gen', 'Col', 'Capt', 'Lt',
            'St', 'vs', 'U.S', 'U.K', 'e.g', 'i.e', 'approx', 'dept', 'govt', 'corp', 'inc'
        ]
        clean_norm = normalized
        for ab in abbrs:
            clean_norm = re.sub(rf'\b{re.escape(ab)}\.', f'{ab}__DOT__', clean_norm, flags=re.IGNORECASE)

        # Regex split on sentence boundaries (. ! ?) while keeping quotes together
        raw_splits = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'“‘])', clean_norm)

        cleaned_sentences: List[str] = []
        for segment in raw_splits:
            segment = segment.replace('__DOT__', '.')
            s = segment.strip().strip('"').strip('“').strip('”').strip("'")
            if self.is_valid_factual_claim(s):
                if not s.endswith(('.', '!', '?')):
                    s = s + '.'
                cleaned_sentences.append(s)

        return cleaned_sentences

    def extract_entities_and_signals(self, claim_text: str) -> Dict[str, Any]:
        """Extracts numerical assertions, named entities, and key verbs from a claim."""
        numbers = re.findall(
            r'\b(?:\d+(?:,\d+)*(?:\.\d+)?|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|hundred|thousand|million|billion)\b)\b',
            claim_text,
            re.IGNORECASE
        )

        words = claim_text.split()
        capitalized = [w.strip('.,;:"\'()') for w in words[1:] if w and w[0].isupper() and len(w) > 2]

        has_verb = (
            bool(VERB_PATTERN.search(claim_text)) or
            bool(MULTILINGUAL_VERB_PATTERN.search(claim_text)) or
            bool(NON_LATIN_SCRIPT_PATTERN.search(claim_text))
        )

        return {
            "numbersFound": numbers,
            "entities": capitalized[:4],
            "hasFactualMarker": has_verb or len(numbers) > 0
        }

    def extract_claims(self, article_text: str, max_claims: int = 6, article_type: str = "News Report") -> List[Dict[str, Any]]:
        """
        Stage 1: Content-cleaning.
        Stage 2: Sentence segmentation.
        Stage 3: Verifiable factual claim filtering & structuring.
        Tags claims with appropriate genre and claimType based on article classification.
        """
        if not article_text or not article_text.strip():
            return []

        # Determine claim type based on genre
        if article_type == "Opinion Piece":
            default_claim_type = "Opinion / Perspective Assertion"
        elif article_type == "Historical Analysis":
            default_claim_type = "Historical Assertion"
        elif article_type == "Feature Article":
            default_claim_type = "Narrative Statement"
        else:
            default_claim_type = "Factual Proposition"

        # Step 0: Speaker attribution detection (from quote-cards, interviews, bylines, or dashes)
        speaker = extract_speaker_attribution(article_text)

        # Step 1: Content-cleaning stage
        cleaned_text = self.clean_article_content(article_text)
        if not cleaned_text.strip():
            cleaned_text = article_text.strip()

        # Step 2: Sentence segmentation with compound claim splitting (Requirement 4)
        raw_sentences = self.split_into_sentences(cleaned_text)
        candidate_sentences = []
        for s in raw_sentences:
            if is_generic_philosophical_truism(s):
                continue
            if speaker:
                s = rewrite_first_person_quote(s, speaker)
            sub_splits = split_compound_claim(s)
            for sub in sub_splits:
                if is_generic_philosophical_truism(sub):
                    continue
                candidate_sentences.append((sub, len(sub_splits) > 1, s))

        claims: List[Dict[str, Any]] = []
        seen_claims = set()

        for idx, (sentence, is_sub, parent_sent) in enumerate(candidate_sentences):
            norm = sentence.lower().strip()
            if norm in seen_claims:
                continue
            seen_claims.add(norm)

            norm_claim = normalize_claim_text(sentence)
            signals = self.extract_entities_and_signals(norm_claim)

            # Check if candidate sentence is an opinion or subjective statement
            is_opinion = bool(OPINION_SENTENCE_REGEX.search(sentence)) or (article_type == "Opinion Piece")
            claim_type = "Opinion / Perspective Assertion" if is_opinion else default_claim_type

            claim_item = {
                "id": f"claim-{len(claims) + 1}",
                "claimNumber": len(claims) + 1,
                "claim": norm_claim,
                "claimType": claim_type,
                "isOpinion": is_opinion,
                "genre": article_type,
                "entities": signals["entities"],
                "numbers": signals["numbersFound"],
                "isTestableProposition": False if is_opinion else signals["hasFactualMarker"],
                "status": "Opinion / Subjective" if is_opinion else "Unverified",
                "isSubClaim": is_sub,
                "parentClaim": parent_sent if is_sub else None,
                "supportingSources": [],
                "contradictingSources": [],
                "supportingEvidenceCount": 0,
                "contradictingEvidenceCount": 0,
                "confidence": 85 if is_opinion else 0,
                "analysisNotes": "Subjective viewpoint assertion (excluded from factual fact-checking)." if is_opinion else ""
            }
            claims.append(claim_item)

            if len(claims) >= max_claims:
                break

        # Fallback only if candidate text itself is a valid factual claim
        if not claims and article_text.strip():
            candidate = article_text.strip()
            if is_generic_philosophical_truism(candidate):
                return []
            if speaker:
                candidate = rewrite_first_person_quote(candidate, speaker)
            cand_to_test = candidate
            cand_clean = self.clean_article_content(candidate)
            if cand_clean and self.is_valid_factual_claim(cand_clean):
                cand_to_test = cand_clean
            elif not self.is_valid_factual_claim(cand_to_test):
                cand_single = " ".join([l.strip() for l in (cand_clean or candidate).splitlines() if l.strip()])
                if self.is_valid_factual_claim(cand_single):
                    cand_to_test = cand_single

            sub_candidates = split_compound_claim(cand_to_test)
            for sub_cand in sub_candidates:
                if is_generic_philosophical_truism(sub_cand):
                    continue
                if speaker:
                    sub_cand = rewrite_first_person_quote(sub_cand, speaker)
                if self.is_valid_factual_claim(sub_cand) or (len(sub_cand.split()) >= 3 and not any(re.search(pat, sub_cand, re.IGNORECASE) for pat in UI_NOISE_PATTERNS)):
                    norm_cand = normalize_claim_text(sub_cand)
                    signals = self.extract_entities_and_signals(norm_cand)
                    claims.append({
                        "id": f"claim-{len(claims) + 1}",
                        "claimNumber": len(claims) + 1,
                        "claim": norm_cand,
                        "claimType": default_claim_type,
                        "genre": article_type,
                        "entities": signals["entities"],
                        "numbers": signals["numbersFound"],
                        "isTestableProposition": True,
                        "status": "Unverified",
                        "isSubClaim": len(sub_candidates) > 1,
                        "parentClaim": cand_to_test if len(sub_candidates) > 1 else None,
                        "supportingSources": [],
                        "contradictingSources": [],
                        "supportingEvidenceCount": 0,
                        "contradictingEvidenceCount": 0,
                        "confidence": 0,
                        "analysisNotes": ""
                    })

        return claims


claim_engine = ClaimExtractionEngine()


def get_claim_engine() -> ClaimExtractionEngine:
    return claim_engine
