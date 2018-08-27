model.bin: model.h5
	./node_modules/keras-js/python/encoder.py -q $<

model.h5:
	python main.py

paper.png: box.png
	convert -size 360x360 xc: png:- |\
	composite -tile box.png - $@

box.png:
	convert -size 36x36 xc: -fill white -stroke black -draw 'rectangle 4,4 31,31' $@