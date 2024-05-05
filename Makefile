el ?=1e-2
db ?= wine
default :
	python melm.py -ds datasets/wine_quality/winequality-white.csv -ty 0 -nh 10  -af dilation -el $(el) -db $(db)

uci:
	python melm.py -ds datasets/uci/housing_scale.txt -ty 0 -nh 10  -af dilation -el $(el) -db uci

