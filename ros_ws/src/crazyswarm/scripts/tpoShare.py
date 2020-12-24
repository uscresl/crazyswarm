#!/usr/bin/env python

import numpy as np
from pycrazyswarm import *
from multiprocessing import Process, Queue
import os 

def p1():
    print('Current pid: {}'.format(os.getpid()))

    #cf.cmdVelocityWorld()

def p2(que):
    print('Current pid: {}'.format(os.getpid()))

    #read vicon
    #simulate KF

    #push q2
    #que.put()


def p3(que):
    print('Current pid: {}'.format(os.getpid()))

    #read q2
    #que.get()

    #if sim bad:
    #   opt
    #   send goTo


def main():
    qCmd = Queue()
    q1 = Queue()
    process1=Process(target=p1,args=(,))
    process2=Process(target=p2,args=(q2,))
    process3=Process(target=p3,args=(q2,))
    process1.start()
    process2.start()
    process3.start()

main()