"""Check LaTeX prose against the authors' style rules.

usage: python3 style_check.py FILE [first_line last_line]   (rules: docs/WRITING_RULES.md)
Rules: no colon, no semicolon, at most two commas per sentence (commas inside \\cite{}, math
and command arguments are ignored). Prints each offending sentence and a count.
"""
import re
import sys


def strip(text):
    text = re.sub(r"(?<!\\)%.*", "", text)                         # comments
    text = re.sub(r"\\begin\{(equation|align|tabular|tikzpicture)\*?\}.*?\\end\{\1\*?\}", " ",
                  text, flags=re.S)
    text = re.sub(r"\$[^$]*\$", "X", text)                          # inline math
    text = re.sub(r"\\(cite|ref|eqref|label|url|emph|textbf|textit)\{([^{}]*)\}",
                  lambda m: m.group(2) if m.group(1) in ("emph", "textbf", "textit") else "R",
                  text)
    text = re.sub(r"\\(begin|end)\{[^{}]*\}(\[[^\]]*\])?", " ", text)
    text = text.replace("Fig.~", "Fig ").replace("et al.", "et al").replace("~", " ")
    text = re.sub(r"\\[a-zA-Z]+\*?", " ", text)
    return text


def main():
    path = sys.argv[1]
    lines = open(path).read().split("\n")
    lo, hi = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (1, len(lines))
    chunk = "\n".join(lines[lo - 1:hi])
    clean = strip(chunk)
    sentences = re.split(r"(?<=[.?!])\s+(?=[A-Z(])", " ".join(clean.split()))
    bad = 0
    for s in sentences:
        issues = []
        if ":" in s:
            issues.append("colon")
        if ";" in s:
            issues.append("semicolon")
        n = s.count(",")
        if n > 2:
            issues.append(f"{n} commas")
        if issues:
            bad += 1
            print(f"[{', '.join(issues)}] {s[:150]}{'...' if len(s) > 150 else ''}")
    print(f"{len(sentences)} sentences, {bad} break the rules")


main()
