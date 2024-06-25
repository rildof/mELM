db ?= wine
default :
	python melm.py -tr benignos_v2.csv -ts malignos_v2.csv -ty 1 -nh 100 -af dilation

mat_db:
	python melm.py -tr matlab/benign_matlab.csv -ts matlab/malign_matlab.csv -ty 1 -nh 10  -af dilation

