list=[


]

l=len(list)


list2=[

]

l2=len(list)


import numpy as np

for i in range(l):

    center = list[i]

    center2 = list2[i]

    random_int = np.random.randint(2, 3)

    result = (center+center2)/random_int

    # mean = center
    # random_int = center / 35
    # if center < 10:
    #     random_int = np.random.randint(1, 2)
    # elif center > 10 and center < 20:
    #     random_int = center / 8
    # elif center > 20 and center < 100:
    #     random_int = center / 15

    # std_dev = random_int
    #
    # simulated_data = np.random.normal(loc=mean, scale=std_dev, size=1)
    print(result)
    # print(result+random_int)