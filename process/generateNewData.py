import Parament

if __name__ == '__main__':
    f = open(Parament.dataPath)
    path = Parament.dataPath.split(".")
    pathNew = path[0] + "double." + path[1]
    fNew = open(pathNew, 'w')
    lines = f.readlines()
    for line in lines:
        fNew.write(str(line))
        fNew.write(str(line))
    f.close()
    fNew.close()
