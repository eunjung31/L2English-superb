import os
from collections import defaultdict

path = "file_path"

fileList = []
for file in os.listdir(path):
    fileList.append(file)

def read_first(filePath):
    firstList, secondList = [], []
    with open(filePath, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            first = line.split("\t")[0]
            second = line.split("\t")[1]
            firstList.append(first)
            secondList.append(second)
    return firstList, secondList

def spk2age(filePath):
    RspkList, RsecondList = read_first(filePath)
    PspkList = []
    for file in fileList:
        spk = file.split("_")[1]
        spk = spk[1:5]
        nspk = file.split("_")[0] + "_" + spk
        PspkList.append((nspk, spk))

    with open(part + '/spk2age', 'w') as f:
        for nspk, spk in PspkList:
            if spk in RspkList:
                index = RspkList.index(spk)
                f.write(f'{nspk}\t{RsecondList[index]}\n')

def spk2gender(filePath):
    RspkList, RsecondList = read_first(filePath)
    PspkList = []
    for file in fileList:
        spk = file.split("_")[1]
        spk = spk[1:5]
        nspk = file.split("_")[0] + "_" + spk
        PspkList.append((nspk, spk))

    with open(part + '/spk2gender', 'w') as f:
        for nspk, spk in PspkList:
            if spk in RspkList:
                index = RspkList.index(spk)
                f.write(f'{nspk}\t{RsecondList[index]}\n')

def text(filePath):
    RspkList, RsecondList = read_first(filePath)
    PspkList = []
    for file in fileList:
        spk = file.split("_")[1].split(".")[0]
        nspk = file.split("_")[0] + "_" + spk
        PspkList.append((nspk, spk))

    with open(part + '/text', 'w') as f:
        for nspk, spk in PspkList:
            if spk in RspkList:
                index = RspkList.index(spk)
                f.write(f'{nspk}\t{RsecondList[index]}\n')

def utt2spk():
    with open(part + '/utt2spk', 'w') as f:
        for fn in fileList:
            spk = fn.split("_")[1]
            spk = spk[1:5]
            nspk = fn.split("_")[0] + "_" + spk
            utt = fn.split(".")[0]
            f.write(f'{utt}\t{nspk}\n')

def wav():
    with open(part + '/wav.scp', 'w') as f:
        for r, d, files in os.walk(path):
            for fn in files:
                wavName = fn.split(".")[0]
                path = os.path.join(r, fn)
                f.write(f'{wavName}\t{path}\n')

def sort_file_by_first_column(file_path): 
    with open(file_path, 'r') as f: 
        lines = f.readlines()
        lines.sort(key=lambda line: line.split('\t')[0])

    with open(file_path, 'w') as f:
        f.writelines(lines)

spk2age("../data2/test/spk2age")
spk2gender("../data2/test/spk2gender")
text("../data2/test/text")
utt2spk()
wav()

sort_file_by_first_column(part + '/spk2age')
sort_file_by_first_column(part + '/spk2gender')
sort_file_by_first_column(part + '/text')
sort_file_by_first_column(part + '/utt2spk')
sort_file_by_first_column(part + '/wav.scp')



