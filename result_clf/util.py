
def ner_ann_to_lines(jdata):

    def overlaps(a, b, c, d):
        if b <= c:
            return -1
        if d <= a:
            return 1
        else:
            return 0

    txt = jdata["text"]
    lbl = jdata["label"]
    lbl_idx = 0
    lines = []
    i = 0
    i_start = 0
    while i < len(txt):
        if i == len(txt)-1 or txt[i] == '\n':
            i = i+1
            while lbl_idx < len(lbl) and (ov:=overlaps(lbl[lbl_idx][0], lbl[lbl_idx][1], i_start, i)) == -1:
                lbl_idx += 1
            if ov == 0:
                lines.append({"text":txt[i_start:i], "label":1})
            else:
                lines.append({"text":txt[i_start:i], "label":0})
            i_start = i
        else:
            i = i+1
    return lines