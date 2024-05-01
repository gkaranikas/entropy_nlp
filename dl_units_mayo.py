from bs4 import BeautifulSoup
import requests
import csv


url = "https://www.mayocliniclabs.com/test-catalog/appendix/measurement"
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0'
}
response = requests.get(url, headers=headers)
response.raise_for_status()

soup = BeautifulSoup(response.content, "html.parser")
table = soup.find("table", attrs={"class":"table table-bordered table-striped"})

if table:
    data = []
    rows = table.find_all("tr")
    for row in rows:
        row_data = []
        cells = row.find_all("td")
        if len(cells) == 2:
            row_data = {
                "Unit": cells[0].text.strip(),
                "Abbreviation": cells[1].text.strip()
            }
            data.append(row_data)

    with open('Units.csv', 'w') as f:
        w = csv.DictWriter(f, ["Unit", "Abbreviation"])
        w.writeheader()
        for row in data:
            w.writerow(row)

else:
    print("Table not found")