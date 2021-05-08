def getCol():
    colstr = []
    for i in range(1,785):
        tmpstr = "d"+str(i)
        colstr.append(tmpstr)
    return colstr
def write(filename,colstr):
    with open(filename, 'a') as f:
        for i in colstr:
            f.write(i+"\n")

if __name__ == '__main__':
    colstr = getCol();
    write("a.txt",colstr)