import os

directory = os.listdir('bin_9')
with open('serialized_rotations11.txt', 'w+') as write_to:
    for file in directory:
        with open('bin_9/' + file , 'r') as read_file:
            a = read_file.readline()
            write_to.write(file[:-4] + ': ' + str(int(a.rstrip().split(",")[1])) + '\n')