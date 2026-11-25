with open("btfr_and_moment0.py") as f:
    s = f.read()

s = s.replace("td = r_des\n", "td = r_des[0]\n")
s = s.replace('td["A_IMAGE"]', 'td["A_IMAGE"][0]')
s = s.replace('td["B_IMAGE"]', 'td["B_IMAGE"][0]')
s = s.replace('td["KRON_RADIUS"]', 'td["KRON_RADIUS"][0]')
s = s.replace("t = r_2m\n", "t = r_2m[0]\n")
s = s.replace('t["r_k20fe"]', 't["r_k20fe"][0]')
s = s.replace('t["k_ba"]', 't["k_ba"][0]')
s = s.replace("top = sf.iloc\n", "top = sf.iloc[0]\n")
s = s.replace("ax1.plot(, ,", "ax1.plot([0], [0],")
s = s.replace("ax2.set_xticks()", "ax2.set_xticks([100, 150, 200, 250, 300, 400])")

with open("btfr_and_moment0.py", "w") as f:
    f.write(s)
