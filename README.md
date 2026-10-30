# Star-forming galaxies + rotation curves & DM in cosmic filaments

Bachelor thesis / final-year project by Ila and Ilya.
Supervisor: Prof. Zamri Zainal Abidin, University of Malaya

## Overview
We use radio observations from the Australia Telescope Compact Array (ATCA)
to:
1. study how galaxies form stars.
2. study how matter is distributed inside galaxies.

## Method
1. Reduce and calibrate ATCA data in MIRIAD
2. Image the field and identify radio sources
3. Crossmatch with an optical galaxy catalogue (VizieR), considering AGN contamination
4. Select galaxies associated with filaments
5. Estimate SFRs using a radio-SFR calibration from the literature
6. Analyse and interpret the results

## Data
Data are **not stored in this repo** (too large). To reproduce:

| Project code | Description | Status |
|--------------|-------------|--------|
| C2767 | <field/target> | in progress |
| C2613 | <field/target> | in progress |

Download from the ATCA archive (ATOA) and place the raw files in
`data/<project>_raw/`. Converted MIRIAD datasets go in `data/<project>_uv/`.

## Setup
Developed on Windows with WSL (Ubuntu 24.04).

**MIRIAD** runs from a Singularity container (`piip.img`), which is not in this repo.
Put it in `~/software/` and add this alias to `~/.bashrc`:

    alias mir='singularity exec ~/software/piip.img'

Then run MIRIAD tasks as e.g. `mir atlod ...`

**Python**

    python3 -m venv ~/fyp
    source ~/fyp/bin/activate
    pip install -r requirements.txt

## Repo structure
    scripts/    reduction and analysis scripts
    notebooks/  exploratory analysis
    notes/      meeting notes, methodology decisions, reduction log
    figures/    plots worth keeping
    data/       local only (gitignored)

## Workflow
- Work on a branch, open a pull request, and have the other person glance at it before merging to `main`
- Pull before you start working: `git pull`
- Log reduction decisions (what was flagged and why) in `notes/reduction-log.md`

## References
- alamak too many
