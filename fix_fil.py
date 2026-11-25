with open("plot_filament_structure.py") as f:
    s = f.read()

s = s.replace("t_wide = res.to_pandas()", "t_wide = res.to_pandas()")
s = s.replace("spine_dir = vt / np.linalg.norm(vt)", "spine_dir = vt / np.linalg.norm(vt)")
s = s.replace("perp_dir = np.array([-spine_dir, spine_dir])", "perp_dir = np.array([-spine_dir, spine_dir])")
s = s.replace("spine_ra = ra0 + (center_xy + t_line * spine_dir) / cos_dec0", "spine_ra = ra0 + (center_xy + t_line * spine_dir) / cos_dec0")
s = s.replace("spine_dec = dec0 + (center_xy + t_line * spine_dir)", "spine_dec = dec0 + (center_xy + t_line * spine_dir)")

with open("plot_filament_structure.py", "w") as f:
    f.write(s)
