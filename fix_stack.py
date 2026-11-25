with open("stack_z1_hi.py") as f:
    code = f.read()

old = "interp_flux = np.interp(v_grid, dv_rest[::-1], spec_mJy[::-1], left=np.nan, right=np.nan)"
new = "order = np.argsort(dv_rest)\n    interp_flux = np.interp(v_grid, dv_rest[order], spec_mJy[order], left=np.nan, right=np.nan)"
assert old in code
with open("stack_z1_hi.py", "w") as f:
    f.write(code.replace(old, new))
