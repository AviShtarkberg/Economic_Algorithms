import cvxpy

""""
This function aims to maximize the product of the utilities of 2 player where each player has different preferences.
At this problem we consider 2 players with preferences that are represented as follows:
Resources:|steal|oil
--------------------
ami       |  1  | 0
--------------------
tami      |  t  | 1-t
Where t is a parameter that lies in [0,1].
For clarity our objective is to maximize(U(a) * U(t)) where U(i) represents the utility of player i.
As we cant insert the multiplication problem to the cvpxy.Problem function we instead insert the sum 
of the logs of U(a) and U(t) as maximizing this function will return a solution that is similar to the 
multiplication problem (As the log function is monotonically increasing).
"""
def max_multiplication_allocation(t):
#####Tests#####
    """
    #Test1: t = 0.5, I expect that ami will receive all the steal and tami will receive all the oil
    >>> result1 =max_multiplication_allocation(0.5)
    >>> round(result1[0],2)
    1.0
    >>> round(result1[1],2)
    0.0

    #Test2: t = 0.75, as I calculated in the pdf we expect that ami will receive 2/3 of the steal,
    #Tami will receive 1/3 of the steal and all the oil.
    >>> result2 =max_multiplication_allocation(0.75)
    >>> round(result2[0],2)
    0.67
    >>> round(result2[1],2)
    0.0

    #Test3: t = 0.25, as I calculated in the pdf we expect that ami will receive all the steal
    #and tami will receive all the oil as t<1/2
    >>> result3 =max_multiplication_allocation(0.25)
    >>> round(result3[0],2)
    1.0
    >>> round(result3[1],2)
    0.0

    #Test4: t = 1, I expect that ami will receive 1/2 of the steal and tami will receive 1/2 of the steal
    # we don't care about the oil as ami and tami referes an 0 value for it.
    >>> result4 =max_multiplication_allocation(1)
    >>> round(result4[0],2)
    0.5

    #Test5: t = 0, I expect that all the steal will be allocated to ami and all the oil to tami.
    >>> result5 =max_multiplication_allocation(0)
    >>> round(result5[0],2)
    1.0
    >>> round(result5[1],2)
    0.0
    """

#####Function#####

    #Define decision variables for ami
    xs, xo = cvxpy.Variable(2)

    #Define utility functions
    utility_ami = 1 * xs
    utility_tami = (1 - xs) * t + (1 - xo) * (1 - t)

    constraints = [xs >= 0, xs <= 1,
                   xo >= 0, xo <= 1]

    #Define and solve the problem
    objective = cvxpy.log(utility_ami) + cvxpy.log(utility_tami)
    prob = cvxpy.Problem(cvxpy.Maximize(objective), constraints)
    prob.solve()

    #Return the optimal values
    return [xs.value, xo.value]


def main():
    return 1

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    main()
