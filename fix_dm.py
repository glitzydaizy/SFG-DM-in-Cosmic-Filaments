with open("fit_dm_halo_z1.py") as f:
    s = f.read()

s = s.replace("t = r_pgc\n        if", "t = r_pgc[0]\n        if")
s = s.replace('t["logd25"]', 't["logd25"][0]')
s = s.replace('t["logr25"]', 't["logr25"][0]')
s = s.replace('r_wise.colnames', 'r_wise[0].colnames')
s = s.replace('r_wise["W1mag"]', 'r_wise[0]["W1mag"][0]')
s = s.replace('popt[', 'popt[2')
s = s.replace('bounds=([0, -300, 80],)', 'bounds=([0, -300, 80], [10, 300, 600])')

with open("fit_dm_halo_z1.py", "w") as f:
    f.write(s)
