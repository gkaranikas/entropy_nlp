import fuzzyset
import regex


def fuzzy_substring_search(major: str, minor: str, errs: int):
    errs_ = 0
    s = None
    while s is None and errs_ <= errs:
        s = regex.search(f"({minor}){{e<={errs_}}}", major)
        errs_ += 1
    return s

def drop_numbers(s):
    words = s.split()
    good_words = []
    for word in words:
        try:
            float(word)
        except ValueError:
            good_words.append(word)
            continue
    return " ".join(good_words)

def probable_number(word):
    word = word.replace("O", "0")
    try:
        return float(word)
    except ValueError:
        return None


class ResultsToFieldsRB:
    def __init__(self, terms, term_match_threshold=0.1):
        self.threshold = term_match_threshold
        self._fs = fuzzyset.FuzzySet()
        for item in terms:
            self._fs.add(item["Abbreviation"])
            for syn in item["Synonyms"]:
                self._fs.add(syn)

    def init_tags(self, lines):
        # tags: O=none, R=range, T=term, V=value, U=units
        # only used for R currently
        return [["O"] * len(l) for l in lines]
    
    def normrange_finder(self, lines, labels, tags):
        npat = r'(\d*\.?\d+|\d+\.?\d*)'
        pattern = f'(?r)({npat}\s*-\s*{npat})|([<>]\s*{npat})'
        for line, label, tag in zip(lines, labels, tags):
            if label["label"] == "LABEL_0":
                continue
            m = fuzzy_substring_search(line, pattern, 0)
            if m:
                start, end = m.span()
                for i in range(start, end):
                    tag[i] = "R"

    def term_finder(self, lines, labels, tags):
        terms = []
        for line, label, tag in zip(lines, labels, tags):
            if label["label"] == "LABEL_0":
                terms.append("")
                continue

            range_start = "".join(tag).find("R")
            sub_line = line[:range_start]
            sub_line = drop_numbers(sub_line)
            result = self._fs.get(sub_line)
            if result is not None and len(result) > 0:
                score, term = result[0]
                if float(score) >= self.threshold:
                    terms.append(term)
                else:
                    terms.append("")
            else:
                terms.append("")
        return terms
    
    def values_finder(self, lines, labels, tags):
        values = []
        for line, label, tag in zip(lines, labels, tags):
            if label["label"] == "LABEL_0":
                values.append([])
                continue

            range_start = "".join(tag).find("R")
            range_end = "".join(tag).rfind("R")
            if range_end != -1:
                range_end += 1

            sub_line1 = line[:range_start]
            sub_line2 = line[range_end:]
            line = sub_line1 + " " + sub_line2
            line_chars = list(line)
            for i, ch in enumerate(line_chars):
                if ch not in {".", ",", "-", "_"} and not ch.isalnum():
                    line_chars[i] = " "
            line = "".join(line_chars)
            words = line.split()
            cur_values = []
            for i, w in enumerate(words):
                if i == 0:
                    continue
                if probable_number(w) is not None:
                    cur_values.append(w)
            values.append(cur_values)
        return values

    def __call__(self, lines, labels):
        tags = self.init_tags(lines)
        self.normrange_finder(lines, labels, tags)
        terms = self.term_finder(lines, labels, tags)
        values = self.values_finder(lines, labels, tags)
        return {
            "term": terms,
            "values": values,
        }
