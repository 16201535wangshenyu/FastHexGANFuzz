import os

from subprocess import Popen, PIPE

bot1 = Popen(["gnome-terminal", "-e", "python FrameSenderReciver_Main_Classification1.py"], stdout=PIPE, stderr=PIPE, stdin=PIPE)
bot2 = Popen(["gnome-terminal", "-e", "python FrameSenderReciver_Main_Classification2.py"], stdout=PIPE, stderr=PIPE, stdin=PIPE)
bot3 = Popen(["gnome-terminal", "-e", "python FrameSenderReciver_Main_Classification3.py"], stdout=PIPE, stderr=PIPE, stdin=PIPE)
bot2 = Popen(["gnome-terminal", "-e", "python FrameSenderReciver_Main_Classification4.py"], stdout=PIPE, stderr=PIPE, stdin=PIPE)
bot3 = Popen(["gnome-terminal", "-e", "python FrameSenderReciver_Main_Classification5.py"], stdout=PIPE, stderr=PIPE, stdin=PIPE)
# os.system("python FrameSenderReciver_Main_Classification1.py")
# os.system("python FrameSenderReciver_Main_Classification2.py")
# os.system("python FrameSenderReciver_Main_Classification3.py")
# os.system("python FrameSenderReciver_Main_Classification4.py")
# os.system("python FrameSenderReciver_Main_Classification5.py")