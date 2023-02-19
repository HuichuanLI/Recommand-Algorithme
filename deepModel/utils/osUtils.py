import json

def readTriple(path,sep=None):
    with open(path,'r',encoding='utf-8') as f:
        for line in f.readlines():
            if sep:
                lines = line.strip().split(sep)
            else:
                lines=line.strip().split()
            if len(lines)!=3:continue
            yield lines

def readFile(path,sep=None):
    with open(path,'r',encoding='utf-8') as f:
        for line in f.readlines():
            if sep:
                lines = line.strip().split(sep)
            else:
                lines = line.strip().split()
            if len(lines)==0:continue
            yield lines

def getJson(path):
    with open(path,'r',encoding='utf-8') as f:
        d=json.load(f)
    return d

def dumpJson(obj,path):
    with open(path,'w+',encoding='utf-8') as f:
        json.dump(obj,f)