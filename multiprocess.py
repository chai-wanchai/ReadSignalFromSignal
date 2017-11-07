from multiprocessing import Process, Lock, Value, Array
import datetime as T
import Preprocess as PP
import ListFile as L
import multiprocessing as ms

from multiprocessing import Process, Queue
from multiprocessing import Pool, TimeoutError
import time
import os

def f(x):
    return x*x

if __name__ == '__main__':
    h=[]
    for i in range(20):
        p = Process(target=f,args=(i,))

        h.append(p)
        p.start()
        print(h)


'''
def f(l, i):
    l.acquire()
    print('hello world', i)
    l.release()

def test():
    lock = Lock()
    s = T.datetime.now()
    for num in range(100):
        Process(target=f, args=(lock, num)).start()

    e = T.datetime.now()
    print(e-s)

def ff(n, a):
    n.value = 3.1415927
    for i in range(len(a)):
        a[i] = -a[i]

if __name__ == '__main__':
    num = Value('d', 0.0)
    arr = Array('i', range(10))

    p = Process(target=ff, args=(num, arr))
    p.start()
    p.join()

    
    print(num.value)

    print(arr[:])
'''
