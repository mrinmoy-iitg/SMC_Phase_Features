#!/bin/bash

for rho in {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1}; do
	for gamma in {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1}; do
		echo $rho $gamma
		python Classification.py $rho $gamma
	done
done

