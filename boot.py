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
lcd.draw_string(100,150,"Done            ")

sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.set_windowing((224, 224))
sensor.run(1)

clock = time.clock()
img_lcd = image.Image()
while True:
    img=sensor.snapshot()
	clock.tick()
	fmap = kpu.forward(task, img)
	lcd.draw_string(0,10,"yay")
	a=img_lcd.draw_image(img,0,0,224,224)
	lcd.draw_string(0,20,"cool")
	fps=clock.fps()
	plist=fmap[:]
	pmax=max(plist)
	max_index=plist.index(pmax)
	a=img_lcd.draw_string(0, 224, "%.2f:%s                            "%(pmax, labels[max_index].strip()))
	a=img_lcd.draw_string(0,20,fps)
	lcd.display(img_lcd)

kpu.deinit(task)
img_lcd.deinit()
