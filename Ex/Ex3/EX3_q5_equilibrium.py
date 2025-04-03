import cvxpy
import numpy as np

"""
    Computes the allocation by maximizing the sum of the weighted logs of the utilities.
"""
def compute_the_allocation(matrix_of_preferences, budgets):
    # Check that there is no negative value in the matrix
    for row in matrix_of_preferences:
        for value in row:
            if value < 0:
                raise ValueError("Preference matrix cannot contain negative values")

    if len(matrix_of_preferences) != len(budgets):
        raise ValueError("Number of players must match the number of budgets")

    num_of_players = len(matrix_of_preferences)
    num_of_resources = len(matrix_of_preferences[0])
    x = cvxpy.Variable((num_of_resources, num_of_players))
    utilities = []

    # Set up each player's utility based on their preferences
    for i in range(num_of_players):
        utility_i = 0
        for j in range(num_of_resources):
            utility_i += x[j, i] * matrix_of_preferences[i][j]
        # Weight the log utility by the player's budget
        utilities.append(budgets[i] * cvxpy.log(utility_i))

    # Define constraints:
    constraints = []
    for j in range(num_of_resources):
        constraints.append(cvxpy.sum(x[j, :]) == 1)
        for i in range(num_of_players):
            constraints.append(x[j, i] >= 0)
            constraints.append(x[j, i] <= 1)

    # Define and solve the optimization problem
    prob = cvxpy.Problem(cvxpy.Maximize(cvxpy.sum(utilities)), constraints)
    prob.solve()

    return prob.value, x.value


"""
    Computes the competitive equilibrium prices for the resources.
"""
def compute_prices(matrix_of_preferences, allocation, budgets):
    num_of_players = len(matrix_of_preferences)
    num_of_resources = len(matrix_of_preferences[0])
    prices = []

    for r in range(num_of_resources):
        price = 0
        # Find a player that gets a positive fraction of resource r.
        for i in range(num_of_players):
            if allocation[r][i] > 1e-6: #defulat eps
                total_utility = 0
                for res in range(num_of_resources):
                    total_utility += allocation[res][i] * matrix_of_preferences[i][res]
                if total_utility > 0:
                    price = (budgets[i] * matrix_of_preferences[i][r]) / total_utility #based on the proof from the slides.
                    break
        prices.append(price)
    return prices


"""
    Computes the competitive equilibrium given a matrix of preferences and budgets.
    It first computes the allocation by maximizing the weighted sum of logs of the utilities,
    then computes the competitive equilibrium prices for the resources.
"""


def competitive_equilibrium(matrix_of_preferences, budgets):
    """
    Testes:
    #Test1 - the example we saw in the class.
    >>> allocation1, prices1 = competitive_equilibrium([[8, 4, 2],[2, 6, 5]],[60,40])
    >>> np.round(allocation1,2)
    array([[1. , 0. ],
           [0.3, 0.7],
           [0. , 1. ]])
    >>> np.round(prices1,1)
    array([52.2, 26.1, 21.7])

    #test2 - equel budgets, as we can see the allocation is envy free.
    >>> allocation2, prices2 = competitive_equilibrium([[8, 4, 2],[2, 6, 5]],[60,60])
    >>> np.round(allocation2,2)
    array([[1., 0.],
           [0., 1.],
           [0., 1.]])
    >>> np.round(prices2,1)
    array([60. , 32.7, 27.3])

    #tset3 - one of the resources is unvaluble for both of the platers and the other 2 are valuable for only 1 player.
    >>> allocation3, prices3 = competitive_equilibrium([[10, 0, 0],[0, 10, 0]],[10,10])
    >>> np.round(allocation3,2)
    array([[1. , 0. ],
           [0. , 1. ],
           [0.5, 0.5]])
    >>> np.round(prices3,1)
    array([10., 10.,  0.])

    #test4- 3 players wgere each has a strong passion for one of the resources:
    >>> allocation4, prices4 = competitive_equilibrium([[10, 1, 2],[3, 8, 3],[1,1,9]],[50,30,20])
    >>> np.round(allocation4,2)
    array([[ 1.,  0., -0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])
    >>> np.round(prices4,1)
    array([50., 30., 20.])

    #test5 - same preferences different bugets
    >>> allocation5, prices5 = competitive_equilibrium([[3, 3, 3],[3, 3, 3]],[50,30])
    >>> np.round(allocation5,2)
    array([[0.62, 0.38],
           [0.62, 0.38],
           [0.62, 0.38]])
    >>> np.round(prices5,1)
    array([26.7, 26.7, 26.7])

    #test6 - different preferences same budgets
    >>> allocation6, prices6 = competitive_equilibrium([[10, 10, 10],[1, 5, 9]],[50,50])
    >>> np.round(allocation6,2)
    array([[1. , 0. ],
           [0.9, 0.1],
           [0. , 1. ]])
    >>> np.round(prices6,1)
    array([26.3, 26.3, 47.4])

    #test7 - negative number error
    >>> competitive_equilibrium([[-10, 20], [30, 40]],[10,10])
    Traceback (most recent call last):
    ...
    ValueError: Preference matrix cannot contain negative values

    #test8 - number of budgets don't match the number of players:
    >>> competitive_equilibrium([[10, 20], [30, 40]],[10,10,10])
    Traceback (most recent call last):
    ...
    ValueError: Number of players must match the number of budgets
    """
    _, allocation = compute_the_allocation(matrix_of_preferences, budgets)
    prices = compute_prices(matrix_of_preferences, allocation, budgets)
    return allocation, prices


def main():
    return


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    main()
