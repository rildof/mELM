db ?= wine
default :
	python melm.py -ds datasets/wine_quality/winequality-white.csv -ty 0 -nh 10  -af dilation -db $(db)

uci:
	python melm.py -ds datasets/uci/housing_scale.txt -ty 0 -nh 10  -af dilation -db uci

