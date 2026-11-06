# C2767 reduction log (2026-11-06)

Raw file: 2012-12-13_0541.C2767 (about 4 GB, not in the repo).

## 1. Copy the raw file from the server (run on the laptop)

    mkdir -p raw_data/zone-1
    scp -p <user>@<server>:<path>/zone-1/raw_data/2012-12-13_0541.C2767 raw_data/zone-1/

## 2. Open MIRIAD (it only exists inside the container)

    cd ~
    singularity shell ~/software/piip.img

## 3. Load into MIRIAD format (inside the container)

    atlod in=$HOME/Projects/FYP-StarFormation/data/C2767_raw/2012-12-13_0541.C2767 out=$HOME/Projects/FYP-StarFormation/data/zone1.uv options=birdie,rfiflag,noauto

Result: 145,320 spectra selected; 8.84% flagged (antenna off-source/off-line); no CACAL data detected. No errors.

## 4. Inspect

    uvindex vis=$HOME/Projects/FYP-StarFormation/data/zone1.uv

- 66,240 records, 2.17 hours, observed 2012-12-13
- 2049 channels per IF, 1 MHz spacing, 3.124 GHz down to about 1.076 GHz (16cm band), 2 IFs
- Polarisations: XX, YY, XY, YX
- Primary calibrator: 1934-638. Phase calibrator: 0302-623
- Targets: zone1field_15 to zone1field_28 (14 mosaic fields)

## 5. Next steps (flagging)

1. Back up: cp -r zone1.uv zone1_backup.uv
2. Plot the 1934-638 spectrum with uvspec, note the bad channel ranges
3. Flag them with uvflag, then pgflag on calibrators and targets
4. Calibrate (mfcal, gpcal, gpboot, uvcat), then flag again
