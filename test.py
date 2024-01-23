
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0
print(f"language: {detect('今一はお前さん')}")