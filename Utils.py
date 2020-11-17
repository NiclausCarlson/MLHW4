from os import listdir

import Letter
import os


def getPaths():
    path = "data/"
    paths = []
    for i in range(10):
        paths.append(path + "part" + str(i + 1))
    return paths


def readLetter(file, isSpam):
    f = open(file, 'r')
    subjectLine = f.readline().split()
    line = f.readline()
    letterLine = f.readline().split()
    subjectLine.remove("Subject:")

    subject = [int(x) for x in subjectLine]
    letter = [int(x) for x in letterLine]
    f.close()

    return Letter.Letter(subject.copy(), letter.copy(), isSpam)


def getParts():
    paths = getPaths()
    parts = []

    for path in paths:
        fileList = listdir(path)
        tmpPart = []
        for file in fileList:
            isSpam = False
            if "spmsg" in file:
                isSpam = True
            tmpPart.append(readLetter(path + "/" + file, isSpam))
        parts.append(tmpPart)

    return parts
