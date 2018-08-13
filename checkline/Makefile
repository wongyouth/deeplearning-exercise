paper.png: box.png
	convert -size 360x360 xc: png:- |\
	composite -tile box.png - $@

box.png:
	convert -size 36x36 xc: -fill white -stroke black -draw 'rectangle 4,4 31,31' $@