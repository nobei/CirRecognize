import os
import re

fileDir = "E:/java/saveDataFromClient/src/audio/"

filePath = [fileDir + 'double click',
            fileDir + 'push',
            fileDir + 'pull',
            fileDir + 'rotation',
            fileDir + 'grab',
            fileDir + 'release',
            fileDir + 'upDown'
            ]

data = open('C:/Users/hao/Desktop/save6-2.txt', 'w')

for i in range(len(filePath)):
    f = os.listdir(filePath[i])
    className = filePath[i].split("/")[-1]
    for j in range(len(f)):
        file = f[j]
        if (re.match(r".*?" + className + ".mat", file, re.M | re.I)):
            time = re.match(r'[0-9]*', file, re.M | re.I)[0]
            data.write(filePath[i] + "/" + file + "-" + className)
            data.write("\n")
