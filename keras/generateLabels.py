import os
import glob

with open('labels.txt', 'w') as f:
	for r, dirs, files in os.walk('../clean-data'):
		for dr in dirs:
			f.write(dr+"\n")
