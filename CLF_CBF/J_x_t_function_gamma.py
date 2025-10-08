#!/usr/bin/env python
# coding: utf-8

# In[2]:



def J_x_t_func(gaussian):
    gamma = 0.6
    if len(gaussian) == 1:
        jxt = 0
    elif len(gaussian) > 1:
        jxt = sum([(gamma**(len(gaussian) - i) * (gaussian[i] + gaussian[i - 1]) / 2) for i in range(1, len(gaussian))])
    else:
        jxt = 0
    return jxt

def J_x_t_func_test():
    Jxt = J_x_t_func([1, 2, 3, 4, 5])
    Jxt_expected = 12.0
    print("Jxt_expected")
    print(Jxt_expected)
    print("Jxt_calculated")
    print(Jxt)

def J_x_t_func_test2():
    Jxt = J_x_t_func([1])
    Jxt_expected = 0.0
    print("Jxt_expected")
    print(Jxt_expected)
    print("Jxt_calculated")
    print(Jxt)

if __name__ == '__main__':
    J_x_t_func_test()
    J_x_t_func_test2()


# In[ ]:




