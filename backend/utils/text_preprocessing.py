import re
import string
import nltk
from typing import List, Dict, Any

# Ensure stopwords are available with fallback
try:
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words('english'))
except Exception:
    STOPWORDS = {
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
        "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
        'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
        'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
        'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
        'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
        'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
        'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
        'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
        'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
        'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
        'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
        'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
        'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll',
        'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't",
        'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't",
        'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',
        "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
        'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
    }

SENSATIONAL_WORDS = {
    'shocking', 'explosive', 'unbelievable', 'secret', 'miracle', 'conspiracy',
    'bombshell', 'illuminati', 'hoax', 'coverup', 'censored', 'exposed',
    'mainstream media won\'t tell you', 'hidden truth', 'panic', 'apocalypse',
    'leaked', 'banned', 'urgent', 'must watch', 'share before deleted'
}

CLICKBAIT_PATTERNS = [
    r'you won\'t believe',
    r'what happens next',
    r'this one trick',
    r'doctors don\'t want you to know',
    r'mind blowing',
    r'will leave you speechless',
    r'number \d+ will shock you'
]

def clean_text(text: str) -> str:
    """
    Standard NLP preprocessing pipeline:
    - Lowercasing
    - Removing URLs, HTML tags, special chars, and digits
    - Removing punctuation
    - Removing stopwords
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Lowercase
    cleaned = text.lower()
    
    # Remove HTML tags
    cleaned = re.sub(r'<.*?>', ' ', cleaned)
    
    # Remove URLs
    cleaned = re.sub(r'https?://\S+|www\.\S+', ' ', cleaned)
    
    # Remove mentions and hashtags
    cleaned = re.sub(r'[@#]\S+', ' ', cleaned)
    
    # Remove punctuation
    cleaned = cleaned.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    cleaned = re.sub(r'\d+', ' ', cleaned)
    
    # Tokenize and remove stopwords
    tokens = cleaned.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    
    return " ".join(tokens)

def analyze_linguistic_signals(raw_text: str) -> Dict[str, Any]:
    """
    Analyzes emotional manipulation, sensationalism, clickbait markers,
    and capitalisation ratios.
    """
    if not raw_text:
        return {
            "sensational_score": 0,
            "clickbait_score": 0,
            "emotional_language": "None",
            "caps_ratio": 0,
            "exclamation_count": 0,
            "detected_sensational_terms": []
        }
    
    lower_text = raw_text.lower()
    
    # Exclamation count
    excl_count = raw_text.count('!')
    
    # Caps ratio
    alpha_chars = [c for c in raw_text if c.isalpha()]
    caps_count = sum(1 for c in alpha_chars if c.isupper())
    caps_ratio = (caps_count / len(alpha_chars)) if alpha_chars else 0
    
    # Sensational keywords detection
    found_sensational = [word for word in SENSATIONAL_WORDS if word in lower_text]
    
    # Clickbait regex patterns
    found_clickbait = [pat for pat in CLICKBAIT_PATTERNS if re.search(pat, lower_text)]
    
    sensational_score = min(100, (len(found_sensational) * 22) + (excl_count * 8) + int(caps_ratio * 50))
    clickbait_score = min(100, (len(found_clickbait) * 35) + (excl_count * 10) + int(caps_ratio * 40))
    
    if sensational_score > 60:
        emotional_language = "High"
    elif sensational_score > 25:
        emotional_language = "Moderate"
    else:
        emotional_language = "None"
        
    return {
        "sensational_score": sensational_score,
        "clickbait_score": clickbait_score,
        "emotional_language": emotional_language,
        "caps_ratio": round(caps_ratio, 3),
        "exclamation_count": excl_count,
        "detected_sensational_terms": found_sensational
    }

def extract_key_claims(raw_text: str, max_claims: int = 4) -> List[str]:
    """
    Extracts key declarative sentences representing claims from raw text.
    """
    if not raw_text:
        return ["No verifiable claim statements could be extracted."]
        
    # Split by sentence terminators
    sentences = re.split(r'(?<=[.!?])\s+', raw_text.strip())
    # Filter for informative sentences (length between 30 and 220 chars)
    claims = [s.strip() for s in sentences if 25 <= len(s.strip()) <= 250]
    
    if not claims:
        # Fallback to chunks
        words = raw_text.split()
        if len(words) > 5:
            claims = [" ".join(words[:25])]
        else:
            claims = [raw_text.strip()]
            
    return claims[:max_claims]
