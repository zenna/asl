# Initalization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
interface = {f.name:f for f in adt.funcs}
push = interface['push'].to_python_lambda(sess)
pop = interface['pop'].to_python_lambda(sess)
zero = adt.consts[0].input_var.eval(sess) # empty data structure
plt.imshow(zero[0,:,:,0]) # initial image of empty


# Look at the first n pushed items and what the n popped items look like
def testPushPop(n, zero, input_data, testname):
    curr_struc = zero
    # Pushing items
    for i in range(n):
        if i <= n-1:
            img = input_data[i:i+1] # may need to shift this index by one?
            plt.imshow(img.reshape(28,28)) # show the image
            plt.savefig('{}_input_{}.png'.format(testname, i+1))
        else: 
            break
        new_struc = push(curr_struc,img)
        plt.imshow(new_struc[0][0,:,:,0])
        plt.savefig('{}_struc_after_input_{}.png'.format(testname, i+1)) # show structure
        curr_struc = new_struc[0]

    
    # Popping items
    for i in range(n):
        (new_struc, output_img)  = pop(curr_struc)
        plt.imshow(new_struc[0,:,:,0])
        plt.savefig('{}_struc_after_output_{}.png'.format(testname, i+1))
        plt.imshow(output_img.reshape(28,28))
        plt.savefig('{}_output_{}.png'.format(testname, i+1))
        curr_struc = new_struc

# testPushPop(2, zero, X_train, 'queue2_n2')


# stack w/ 2: 1488669343.6124659adt_stack__nitems_2__
# stack w/ 5: /home/jackiexu/adt_data/pdt/1488669346.590199adt_stack__nitems_5__/model.ckpt - training more

# queue w/ 2: /home/jackiexu/adt_data/pdt/1488669344.378259adt_queue__nitems_2__/model.ckpt
# queue w/ 5: /home/jackiexu/adt_data/pdt/1488669345.000821adt_queue__nitems_5__/model.ckpt


# Try to push and pop at different rates; TODO
