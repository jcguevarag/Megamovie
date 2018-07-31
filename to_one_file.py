import os

directory = os.listdir(os.listdir('.')[-2])
directory = ['textFiles/' + x for x in directory]

with open('serialized_rotations.txt', 'w+') as write_to:
    for file in directory:
        with open(file , 'r') as read_file:
            a = read_file.readline()
            write_to.write(file[10:-5] + ': ' + str(int(a.rstrip().split(",")[1])) + '\n')