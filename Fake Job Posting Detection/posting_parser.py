"""Split a pasted job posting into company / description / requirements.

Heuristic, rule-based:
  1. Section headings ("About us", "Responsibilities", "Requirements"...) start a new section.
     Text before the first heading is treated as the description.
  2. "Location: ...", "Job Title: ..." style lines are metadata and set aside.
  3. If no company section was found, sentences that read like a company introduction
     ("Acme is a leading provider...", "headquartered in...") are moved into it.
If no recognisable headings exist, `structured` is False so callers shouldn't claim that
the company section is missing.
"""
import re
from dataclasses import dataclass, field

_Q = r"['’]?"  # optional apostrophe

# Order matters: the first matching category wins. Every pattern must match the WHOLE heading line.
_HEADINGS = [
    ("requirements", rf"requirements?|(?:required |minimum |preferred |basic |key )?qualifications?|"
                     rf"what you{_Q}ll (?:need|bring|have)|what you will (?:need|bring|have)|what you need|you have|you(?:{_Q}ve| have) got|who you are|"
                     rf"what we{_Q}re looking for|what we are looking for|skills?(?: (?:and|&) (?:experience|qualifications|requirements))?|"
                     rf"(?:required |technical |key )?skills|must[- ]haves?|nice[- ]to[- ]haves?|bonus(?: points)?|"
                     rf"(?:required )?experience(?: required)?|experience (?:and|&) (?:skills|education)|"
                     rf"education(?: (?:and|&) experience)?|your profile|about you|ideal candidate|candidate profile|"
                     rf"preferred (?:skills|experience)|requirements (?:and|&) qualifications"),
    ("description", rf"job description|(?:key |main |core |primary )?responsibilities|duties|(?:key )?duties(?: (?:and|&) responsibilities)?|"
                    rf"responsibilities (?:and|&) duties|role (?:and|&) responsibilities|the (?:role|job|position|opportunity)|"
                    rf"about the (?:job|role|position|opportunity)|role (?:overview|description)|position (?:summary|description|overview)|"
                    rf"what you{_Q}ll (?:do|be doing)|what you will (?:do|be doing)|your role|your day[- ]to[- ]day|in this role|"
                    rf"(?:job )?overview|(?:job )?summary|description|the opportunity|essential functions|"
                    rf"about this (?:role|job|position|opportunity)|job details|position details"),
    ("company", rf"about (?:us|the company|our company|the team|the organi[sz]ation|company)|"
                rf"about (?-i:[A-Z])[\w&.,\-' ]{{0,55}}|who we are|who is [\w&.,\-' ]{{1,40}}\??|"
                rf"company (?:overview|description|profile|background|information|info|summary)|"
                rf"(?:the |our )?company|our (?:company|story|mission|vision|values|culture|team|history)|"
                rf"overview of the company|a little about us|who are we"),
    ("benefits", rf"benefits?(?: (?:and|&) perks)?|what we offer|perks(?: (?:and|&) benefits)?|compensation(?: (?:and|&) benefits)?|"
                 rf"we offer|why join us|why work (?:with|for) us|what{_Q}s in it for you|our benefits|total rewards|"
                 rf"salary(?: (?:and|&) benefits)?|pay (?:and|&) benefits"),
    ("ignore", rf"equal opportunity.*|how to apply|apply now|about the application|application process"),
]
_COMPILED = [(name, re.compile(rf"^(?:{pat})$", re.I)) for name, pat in _HEADINGS]

# "Requirements: Python, SQL" -> heading + inline content
_INLINE = re.compile(r"^([^:]{2,60}):\s*(\S.*)$")
# one-line metadata that must not switch the current section
_META = re.compile(r"^(?:job )?(?:title|location|job type|employment type|contract type|work type|workplace|"
                   r"salary|pay|pay rate|department|industry|posted|date posted|job id|job ref(?:erence)?|req(?:uisition)? id|"
                   r"schedule|hours|seniority level|job function|reference)\s*:\s*\S.*$", re.I)

# --- company-intro detection (fallback when there is no company heading) -----------------------------
_COMPANY_NOUNS = (r"company|firm|agency|startup|start-up|organi[sz]ation|business|provider|leader|corporation|studio|"
                  r"consultancy|chain|group|team|institution|hospital|school|university|charity|non-?profit|manufacturer|"
                  r"retailer|brand|platform|network|marketplace|bank|practice|clinic|stores|enterprise|developer|supplier|"
                  r"publisher|operator|distributor|laboratory|lab|foundation|partnership")
_CUE_A = re.compile(r"\b(?:headquartered|founded (?:in|by)|established in|our mission|our vision|our (?:clients|customers)|"
                    r"\d[\d,]*\+?\s+(?:employees|staff|team members)|offices? (?:in|across))\b", re.I)
_SUBJECT = re.compile(r"^(?:we|our\s+[\w-]+|the\s+company|[A-Z][\w&.'\-]*(?:\s+[A-Z][\w&.'\-]*){0,4})\s+"
                      r"(?:is|are|was|has been|have been)\s+(?:a|an|the|one of the|among the)\b")
_NOT_SUBJECT = {"this", "it", "there", "you", "that", "these", "the role", "the job", "the position", "the successful"}
_CUE_B = re.compile(rf"\b(?:{_COMPANY_NOUNS})\b", re.I)


@dataclass
class ParsedPosting:
    company: str = ""
    description: str = ""
    requirements: str = ""
    title: str = ""              # from a "Job Title:" line, or a short first line
    benefits: str = ""           # benefits / perks section
    ignored: str = ""            # how to apply, location, pay etc.; not used by the model
    structured: bool = False     # True if at least one heading was recognised
    company_inferred: bool = False  # company text was picked out by sentence cues, not a heading
    found: list = field(default_factory=list)


def _classify(line):
    """Return the section name if `line` is a heading, else None."""
    text = re.sub(r"^[#*\-\s>_•·]+|[*_\s:]+$", "", line.strip())
    if not text or len(text) > 60:
        return None
    for name, rx in _COMPILED:
        if rx.match(text):
            return name
    return None


def _is_company_sentence(sentence):
    s = sentence.strip()
    if _CUE_A.search(s):
        return True
    m = _SUBJECT.match(s)
    if not m:
        return False
    subject = re.match(r"^\S+(?:\s+\S+)?", s).group(0).lower()
    if subject.split()[0] in _NOT_SUBJECT or subject in _NOT_SUBJECT:
        return False
    # the noun must appear soon after "is a/an/the ..." (not buried in a long job sentence)
    tail = " ".join(s[m.end():].split()[:14])
    return bool(_CUE_B.search(tail))


def _pull_company_sentences(text):
    """Split `text` into (company_sentences, remaining_text)."""
    company, rest = [], []
    for line in text.splitlines():
        if not line.strip():
            rest.append("")
            continue
        keep = []
        for sentence in re.split(r"(?<=[.!?])\s+", line.strip()):
            (company if _is_company_sentence(sentence) else keep).append(sentence)
        if keep:
            rest.append(" ".join(keep))
    return "\n".join(company).strip(), "\n".join(rest).strip()


def _first_line_title(description, structured):
    """A short, unpunctuated first line followed by more text is almost always the job title."""
    lines = description.splitlines()
    if len(lines) < 2 or not structured:
        return "", description
    first = lines[0].strip()
    if 2 <= len(first) <= 80 and not re.search(r"[.!?:;,]$", first) and len(first.split()) <= 10:
        return first, "\n".join(lines[1:]).strip()
    return "", description


def parse_posting(raw):
    sections = {"company": [], "description": [], "requirements": [], "benefits": [], "ignore": []}
    meta, title = [], ""
    current, structured, found = "description", False, []

    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            sections[current].append("")
            continue
        clean_line = stripped.lstrip("#*- ")
        if _META.match(clean_line):
            meta.append(clean_line)
            if not title and re.match(r"(?:job )?title\s*:", clean_line, re.I):
                title = clean_line.split(":", 1)[1].strip()
            continue
        heading = _classify(line)
        content = None
        if heading is None:
            m = _INLINE.match(stripped.lstrip("#*- ").rstrip())
            if m:
                inline_heading = _classify(m.group(1))
                if inline_heading:
                    heading, content = inline_heading, m.group(2)
        if heading:
            structured = True
            current = heading
            if heading not in found:
                found.append(heading)
            if content:
                sections[current].append(content)
            continue
        sections[current].append(stripped)

    join = lambda key: "\n".join(sections[key]).strip()
    company, description = join("company"), join("description")
    if not title:
        title, description = _first_line_title(description, structured)
    inferred = False
    if not company:
        company, description = _pull_company_sentences(description)
        inferred = bool(company)
    return ParsedPosting(company=company, description=description, requirements=join("requirements"),
                         title=title, benefits=join("benefits"),
                         ignored="\n".join([x for x in (join("ignore"), "\n".join(meta)) if x]),
                         structured=structured or inferred, company_inferred=inferred, found=found)
