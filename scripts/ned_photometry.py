from astroquery.ipac.ned import Ned

for n in ["WISEA J033704.82-530121.7", "WISEA J033705.94-525820.4"]:
    print("\n", n)
    Ned.get_table(n, table="photometry").pprint(max_width=-1, max_lines=-1)
