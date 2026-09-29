# ============================================================
#  Layer 2 — Feature Extraction  (25 signals)
# ============================================================

import math, re
from urllib.parse import urlparse
import tldextract

PHISHING_WORDS = [
    "login", "signin", "secure", "account", "update", "verify",
    "banking", "confirm", "password", "paypal", "ebay", "amazon",
    "apple", "microsoft", "google", "support", "free", "winner",
    "click", "urgent", "suspend", "validate", "recover", "unusual",
    "activity", "billing", "checkout", "redirect", "authenticate",
]

DODGY_TLDS = {
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "club",
    "online", "site", "info", "biz", "pw", "cc", "ws",
    "guru", "link", "work", "live", "click", "icu",
}

BRANDS = [
    "paypal", "google", "apple", "microsoft", "amazon", "facebook",
    "netflix", "ebay", "instagram", "twitter", "linkedin", "dropbox",
    "chase", "wellsfargo", "bankofamerica", "citibank", "hsbc",
]

IS_IP = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")


def shannon_entropy(text):
    if not text:
        return 0.0
    freq = {}
    for c in text:
        freq[c] = freq.get(c, 0) + 1
    n = len(text)
    return -sum((v/n) * math.log2(v/n) for v in freq.values())


def get_features(url):
    parsed   = urlparse(url if "://" in url else "http://" + url)
    parts    = tldextract.extract(url)
    host     = parsed.hostname or ""
    path     = parsed.path or ""
    query    = parsed.query or ""
    sub      = parts.subdomain or ""
    domain   = parts.domain or ""
    low      = url.lower()

    return {
        "url_length":           len(url),
        "hostname_length":      len(host),
        "dot_count":            host.count("."),
        "hyphen_count":         host.count("-"),
        "slash_count":          path.count("/"),
        "underscore_count":     url.count("_"),
        "subdomain_depth":      len(sub.split(".")) if sub else 0,
        "domain_name_length":   len(domain),
        "uses_https":           1 if parsed.scheme == "https" else 0,
        "uses_ip_address":      1 if IS_IP.match(host) else 0,
        "has_at_symbol":        1 if "@" in url else 0,
        "has_double_slash":     1 if "//" in path else 0,
        "has_hex_encoding":     1 if re.search(r"%[0-9a-fA-F]{2}", url) else 0,
        "has_punycode":         1 if "xn--" in host.lower() else 0,
        "punycode_label_count": host.lower().count("xn--"),
        "has_query_string":     1 if query else 0,
        "query_param_count":    len(query.split("&")) if query else 0,
        "has_fragment":         1 if parsed.fragment else 0,
        "phishing_word_count":  sum(w in low for w in PHISHING_WORDS),
        "digit_ratio":          round(sum(c.isdigit() for c in url) / max(len(url), 1), 4),
        "url_entropy":          round(shannon_entropy(url), 4),
        "dodgy_tld":            1 if (parts.suffix or "").lower() in DODGY_TLDS else 0,
        "brand_in_subdomain":   1 if any(b in sub.lower()  for b in BRANDS) else 0,
        "brand_in_path":        1 if any(b in path.lower() for b in BRANDS) else 0,
        "typosquat_hyphen":     1 if any(f"{b}-" in host.lower() or f"-{b}" in host.lower() for b in BRANDS) else 0,
    }
