import datetime
#import datefinder
from dateparser.search import search_dates


class DateFinder:
    def __init__(self):
        self.settings = {
            'DATE_ORDER': 'DMY',
            'REQUIRE_PARTS' : ['year', 'month'],
            'PREFER_DAY_OF_MONTH': 'first'
        }

    def __call__(self, lines, labels):
        output = []
        for line, label in zip(lines, labels):
            if label["label"] == "LABEL_0":
                # TODO very slow
                matches = search_dates(line, settings=self.settings)
                if matches is not None:
                    dates = [m[1] for m in matches]
                    good_dates = []
                    for dt in dates:
                        if dt > datetime.datetime(1900,1,1):
                            good_dates.append(
                                datetime.datetime(dt.year, dt.month, dt.day)
                            )
                    output.append(good_dates)
                else:
                    output.append([])
            else:
                output.append([])
        return output