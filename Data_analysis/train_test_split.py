import numpy as np
from sklearn.model_selection import train_test_split

def remove_data(data):
    del_ar = []
    i_len = len(data)
    for i in range(i_len):
        # print(data[i][0])
        if data[i][0] < 0.001:
            # print("HERE")
            del_ar.append(i)
    print(len(del_ar))
    data = np.delete(data, (del_ar), axis=0)
    f_len = len(data)
    print("%d, data points removed." % (f_len / i_len))
    return data

x =  np.genfromtxt("mlz_input.txt")
x = remove_data(x)
# print(x)
np.random.shuffle(x) ##########################
# print("\n\n\n")
# print(x)
hl = int(len(x)/ 2)

train = x[: hl]
test = x[hl :]
# print(len(train))
# print(len(test))
# print(len(x))

np.savetxt("train_data.txt", train, fmt='%.6f', header='zs u g r i z u-g g-r r-i i-z eu eg er ei ez eu-g eg-r er-i ei-z')
np.savetxt("test_data.txt", test, fmt='%.6f', header='zs u g r i z u-g g-r r-i i-z eu eg er ei ez eu-g eg-r er-i ei-z')
