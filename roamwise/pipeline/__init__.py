"""Scripts that produce the shipped datasets in `roamwise/data/`.

Run them from this directory -- they import each other by module name:

    cd roamwise/pipeline
    python build_catalogue.py PAR BER      # -> ../data/poi.csv
    python gold_list.py PAR BER --catalogue ../data/poi.csv

`common.CITIES` is the single city registry; adding a destination means adding
a row there rather than editing each script.
"""
