# --------------------------- DATA PREPROCESSING -------------------------------

rule pantheon_hubble_flow:
    input:
        "src/static/Pantheon+SH0ES.dat",
        "src/static/Pantheon_HOSTGAL_LOGMASS.txt",
        "src/static/Pantheon_HOSTGAL_sSFR.txt",
        "src/static/jones_et_al_18.fit",
        "src/static/uddin_et_al_17.fit",
        "src/scripts/reusable_scripts/preprocessing.py",
        "src/scripts/reusable_scripts/configs/preprocessing.yaml"
    cache:
        True
    output:
        directory("src/data/pantheon_hubble_flow")
    script:
        "src/scripts/bash_scripts/pantheon_hubble_flow.sh"