#!/usr/bin/env python

import numpy as np
from pycrazyswarm import *
from multiprocessing import Process, Queue
import os 

def p1(qCmd, queue1):
    print('Current pid: {}'.format(os.getpid()))
    
    #if not qcmd.empty():
        #read cmd
        cmd = qcmd.get()

        #send cmd
        #cf.cmdVelocityWorld()

    #read pos and push to q1
    #queue1.push(cf.position())

def p2(queue1, queue2):
    print('Current pid: {}'.format(os.getpid()))

    #read q1
    #queue1.get()

    #simulate KF

    #push kf to q2
    #queue2.put(kf)


def p3(queue2, qCmd):
    print('Current pid: {}'.format(os.getpid()))

    #read q2
    #queue2.get()

    #if sim bad:
    #   opt
    #   send goTo
    #   qCmd.put()


def main():
    qCmd = Queue()
    q1 = Queue()
    q2 = Queue()

    process1=Process(target=p1,args=(qCmd, q1))
    process2=Process(target=p2,args=(q1, q2))
    process3=Process(target=p3,args=(q2, qCmd))
    process1.start()
    process2.start()
    process3.start()

main()