import os
import re

if __name__ == '__main__':
    dirPath = "C:/Users/hao/Desktop/jiangyi/push"
    for file in os.listdir(dirPath):
        time = re.match(r'[0-9]*', file, re.M | re.I)[0]
        typeName = time+'.*'
        count = 0
        for fileType in os.listdir(dirPath):
            match = re.match(r''+typeName,fileType,re.M | re.I)
            if str(match) != 'None':
                count += 1

        if count != 6:
            print(time+'\n')


