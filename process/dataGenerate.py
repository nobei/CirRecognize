import os
import re

fileDir = ["D:/dataSet/",
           "C:/Users/hao/Desktop/jiangyi/"]

moveType = ['double click',
            'push',
            'pull',
            'rotation',
            'grab',
            'release',
            'upDown'
            ]

data = open('C:/Users/hao/Desktop/save10-27.txt', 'w')
for index in range(len(fileDir)):
    filePath = [fileDir[index] + moveType[i] for i in range(len(moveType))]
    for i in range(len(filePath)):
        f = os.listdir(filePath[i])
        className = filePath[i].split("/")[-1]
        for j in range(len(f)):
            file = f[j]
            if (re.match(r".*?" + className + ".mat", file, re.M | re.I)):
                time = re.match(r'[0-9]*', file, re.M | re.I)[0]
                data.write(filePath[i] + "/" + file + "-" + className)
                data.write("\n")
