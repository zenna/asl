from collections import deque
import copy
class Axiom(object):
    def __init__(self, lhs, rhs, name):
        self.lhs = lhs
        self.rhs = rhs
        self.name = name

def q_ax(empty, push, pop, items):
    nitems = len(items)
    axioms = []
    eqqueue = empty
    eqqueues = [eqqueue]
    for i in range(nitems):
        orig_eqqueue = copy.deepcopy(eqqueue)
        print("orig_q", orig_eqqueue)
        (push_eqqueue,) = push(orig_eqqueue, items[i]) # pushed the item onto the eqqueue
        eqqueues.append(push_eqqueue)
        pop_eqqueue = copy.deepcopy(push_eqqueue)
        print("push_q", push_eqqueue)
        for j in range(i+1):
            # Item equivalence
            (pop_eqqueue, pop_item) = pop(pop_eqqueue) # when you pop item from queue
            print("pop_q", pop_eqqueue)
            axiom = Axiom((pop_item,), (items[j],), 'item-eq%s-%s' %(i, j))
            axioms.append(axiom)

            # Eqqueue equivalence, Case 1: Orig queue was empty
            if i==j:
                axiom = Axiom((pop_eqqueue,), (empty,), 'eqqueue-eq%s-%s' %(i, j))
                axioms.append(axiom)

            # Eqqueue equivalence, Case 2: Orig queue had items
            else:
                (test_pop_eqqueue, test_pop_item) = pop(orig_eqqueue)
                (test_push_eqqueue, ) = push(test_pop_eqqueue, items[i])
                print("test_push_q", test_push_eqqueue)
                print("pop_q (sanitycheck)", pop_eqqueue)
                axiom = Axiom((pop_eqqueue,), (test_push_eqqueue,), 'eqqueue-eq%s-%s' %(i, j)) #queue.push(i)[0].pop()[0] == queue.pop()[0].push(i)[0]
                axioms.append(axiom)
        # Set next queue to support one more item 
        eqqueue=copy.deepcopy(push_eqqueue)
        print()
    return axioms


empty = deque()
def push(q, i):
    new_q = copy.deepcopy(q)
    new_q.append(i)
    return (new_q,)

def pop(q):
    new_q = copy.deepcopy(q)
    i = new_q.popleft()
    return (new_q, i)
    
items = [3,1,5,6]
axioms = q_ax(empty, push, pop, items)
