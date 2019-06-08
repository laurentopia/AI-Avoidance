import sensor, image, lcd, os, KPU as kpu, sys, time

lcd.init(color=(150,70,255))
lcd.freq(16000000)
lcd.direction(lcd.YX_LRUD)

lcd.draw_string(100,96,"Obstacle Detector")
lcd.draw_string(100,120,"launched from "+os.getcwd())

lcd.draw_string(100,150,"Loading labels...")
f=open("labels.txt",'r')
labels=f.readlines()
f.close()
lcd.draw_string(100,150,"Loading model...")
task = kpu.load(0x200000)
lcd.draw_string(100,150,"Loaded            ")

sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.set_windowing((224, 224))
sensor.run(1)

clock = time.clock()
img_lcd = image.Image()
lcd.clear()
while True:
	img=sensor.snapshot()
	clock.tick()
	fmap = kpu.forward(task, img)
	lcd.draw_string(0,10,"DEBUG : forward calculated")
	a=img_lcd.draw_image(img,0,0,224,224)
	lcd.draw_string(0,20,"DEBUG : image drawn")
	fps=clock.fps()
	plist=fmap[:]
	pmax=max(plist)
	max_index=plist.index(pmax)
	a=img_lcd.draw_string(0, 224, "%.2f:%s                            "%(pmax, labels[max_index].strip()))
	a=img_lcd.draw_string(0,20,int(fps))
	# TODO: fix this draw gauges for each label
	for i in range (0,4):
		prob = int(plist[i] * 255)
		a=img_lcd.draw_rectangle(i*32,200-prob/4,30,prob/4, color = (prob, prob, prob), fill=True)
		a=img_lcd.draw_string(i*32,205,labels[i].strip())
	lcd.display(img_lcd)

kpu.deinit(task)
img_lcd.deinit()