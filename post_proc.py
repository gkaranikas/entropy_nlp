
def get_latest_values(fields, dates):
    left_to_right = True
    output = []
    for term, values, date_list in zip(fields["term"], fields["values"], dates):
        if len(date_list) > 0:
            if len(date_list) > 1:
                if date_list[1] > date_list[0]:
                    left_to_right = True
                else:
                    left_to_right = False
            output.append(None)
        elif term != "" and len(values) > 0:
            if len(values) > 1:
                if left_to_right:
                    v = values[-1]
                else:
                    v = values[0]
            else:
                v = values[0]
            output.append(v)
        else:
            output.append(None)
    return output

def remove_duplicates(rows):
    pass


class PostProcess:
    def __init__(self, terms):
        self.synonym_dict = dict()
        for jd in terms:
            abbr = jd["Abbreviation"]
            self.synonym_dict[abbr] = abbr
            for syn in jd["Synonyms"]:
                self.synonym_dict[syn] = abbr

    def __call__(self, lines, labels, fields, dates):
        latest_values = get_latest_values(fields, dates)

        # construct final output, removing duplicates
        output = []
        for term, value in zip(fields["term"], latest_values):
            if term != "" and value is not None:
                item = {
                    "parameter": self.synonym_dict[term],
                    "value": value,
                    "units": ""
                }
                if item not in output:
                    output.append(item)

        return output