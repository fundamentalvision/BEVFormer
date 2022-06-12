import functools
import time
from collections import defaultdict
time_maps = defaultdict(lambda :0.)
count_maps = defaultdict(lambda :0.)
def run_time(name):
    def middle(fn):
        def wrapper(*args, **kwargs):
            start = time.time()
            res = fn(*args, **kwargs)
            time_maps['%s : %s'%(name, fn.__name__) ] += time.time()-start
            count_maps['%s : %s'%(name, fn.__name__) ] +=1
            print("%s : %s takes up %f "% (name, fn.__name__,time_maps['%s : %s'%(name, fn.__name__) ] /count_maps['%s : %s'%(name, fn.__name__) ] ))
            return res
        return wrapper
    return middle
    