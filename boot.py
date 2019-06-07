import sensor, image, lcd, os, KPU as kpu, sys, time

lcd.init(color=(100,50,255))
lcd.freq(16000000)
lcd.direction(lcd.YX_LRUD)

lcd.draw_string(100,96,"Obstacle Detector")
lcd.draw_string(100,120,"launched from "+os.getcwd())

lcd.draw_string(100,150,"Loading labels...")
f=open("/sd/labels.txt",'r')
labels=f.readlines()
f.close()
lcd.draw_string(100,150,"Loading model...")
task = kpu.load("/sd/avoidance.kmodel")

sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.set_windowing((224, 224))
sensor.run(1)

clock = time.clock()
while True:
	img=sensor.snapshot()
	clock.tick()
	fmap = kpu.forward(task, img)
	lcd.display(img)
	fps=clock.fps()
	plist=fmap[:]
	pmax=max(plist)    
	max_index=plist.index(pmax)    
	lcd.draw_string(0, 224, "%.2f:%s                            "%(pmax, labels[max_index].strip()))

a = kpu.deinit(task)