from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

if detect('hola que')=='es':
    print('YEAS')