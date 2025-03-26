import cvxpy
"""
    This function aims to give approximation of the egalitarian allocation problem.
    The function uses the main functionality of cvxpy methods to give such allocation based on the
    preferences of the players that want to share resources between them where they
    have different preferences about the resources.
    """
def egalitarian_allocation(matrix_of_preferences, suppress_print=False):
    """
    The funciton returns a list where:
      ans[0] = "optimal" if the optimization was successful, otherwise "non optimal"
      ans[1] = optimal value
      ans[2] = utility for agent 1
      ans[3] = utility for agent 2
      ... and so on for additional agents

    Tests:
    #test1 - the case we saw in the class:
    >>> result1 = egalitarian_allocation([[80,19,1], [79,1,20]], suppress_print=True)
    >>> result1[0]
    'optimal'
    >>> round(result1[1], 2)
    59.25
    >>> round(result1[2], 2)
    59.25
    >>> round(result1[3], 2)
    59.25

    #test2 - another case we saw in the class(where te allocation is fair but not efficient):
    >>> result2 = egalitarian_allocation([[100,0], [0,50]], suppress_print=True)
    >>> result2[0]
    'optimal'
    >>> round(result2[1], 2)
    50.0
    >>> round(result2[2],2) > 50
    True
    >>> round(result2[3], 3)
    50.0

    #test3 - case where evrey value is set to 0:
    >>> result3 = egalitarian_allocation([[0,0], [0,0]], suppress_print=True)
    >>> result3[0]
    'optimal'
    >>> round(result3[1], 2)
    0.0
    >>> round(result3[2], 2)
    0.0

    #test4 - simple regular test with 2 players:
    >>> result4 = egalitarian_allocation([[100,20], [20,100]], suppress_print=True)
    >>> result4[0]
    'optimal'
    >>> round(result4[1], 2)
    100.0
    >>> round(result4[2], 2)
    100.0
    >>> round(result4[3], 2)
    100.0

    #test5 - test with more than 2 players but simple allocation:
    >>> result5 = egalitarian_allocation([[0,0,3], [0,21,0], [11,0,0]], suppress_print=True)
    >>> result5[0]
    'optimal'
    >>> round(result5[1], 2)
    3.0
    >>> round(result5[2], 2)
    3.0
    >>> round(result5[3], 2) >3
    True
    >>> round(result5[4], 2)>3
    True

    #test6 - test with more than 2 players
    >>> result6 = egalitarian_allocation([[2,4,8], [8,2,4], [2,3,2]], suppress_print=True)
    >>> result6[0]
    'optimal'

    #test7 - 4 players with 2 resources:
    >>> result7 = egalitarian_allocation([[100,0],[0,100],[50,50],[30,30]], suppress_print=True)
    >>> result7[0]
    'optimal'

    #test8 - 3 players with 4 resources:
    >>> result8 = egalitarian_allocation([[100,0,3,4],[0,100,1,2],[50,50,4,5],[30,30,7,7]], suppress_print=True)
    >>> result8[0]
    'optimal'

    #test9 - 2 players with values between 0 and 1:
    >>> result9 = egalitarian_allocation([[0.7, 0.1],[0.3, 0.4]], suppress_print=True)
    >>> result9[0]
    'optimal'

    #test10 - I would like to check that if there is a negative value in the matrix its not working:
    >>> egalitarian_allocation([[-10, 20], [30, 40]], suppress_print=True)
    Traceback (most recent call last):
    ...
    ValueError: Preference matrix cannot contain negative values

    """
   #At first, I will check that there is no negative value in the matrix as the preferences should be None negative!
    for row in matrix_of_preferences:
        for value in row:
            if value < 0:
                raise ValueError("Preference matrix cannot contain negative values")
    # initial settings for the values that we want to use in that problem.
    num_of_resources = len(matrix_of_preferences[0])
    num_of_players = len(matrix_of_preferences)
    x = cvxpy.Variable((num_of_resources, num_of_players))
    utilities = []

    # set up the variables for each player and their preferences.
    for i in range(num_of_players):
        utility_i = 0
        for j in range(num_of_resources):
            utility_i += x[j][i] * matrix_of_preferences[i][j]
        utilities.append(utility_i)

    # variable that we will want to maximize that represent the minimum above the players as learned in the class.
    z = cvxpy.Variable()

    # initial setting of the constraints that defines the problem.
    constraints = []
    for i in range(num_of_resources):
        constraints.append(cvxpy.sum(x[i, :]) == 1)  # the sum of each variable should be 1 so we receive %
        for j in range(num_of_players):
            constraints.append(x[i, j] >= 0)
            constraints.append(x[i, j] <= 1)
    # we want that each utility will be higher than the z that we have set so we receive that z is the min value.
    for i in range(num_of_players):
        constraints.append(utilities[i] >= z)

    # Define the optimization problem: maximize the minimum utility.
    prob = cvxpy.Problem(cvxpy.Maximize(z), constraints)
    # Solve the problem.
    prob.solve()

    if not suppress_print:
        # print all the relevant values that were calculated:
        print("Status:", prob.status)
        print("optimal value:", prob.value)
        for j in range(num_of_players):
            print(f"Utility for Agent {j + 1}:", utilities[j].value)
        print("\n--- Allocation ---\n")
        for player_index in range(num_of_players):
            resource_descriptions = []
            for resource_index in range(num_of_resources):
                fraction = x[resource_index, player_index].value
                resource_descriptions.append(f"{fraction:.2f} of resource #{resource_index + 1}")
            if num_of_resources == 1:
                resources_str = resource_descriptions[0]
            else:
                resources_str = ", ".join(resource_descriptions[:-1])
                resources_str += f", and {resource_descriptions[-1]}"
            print(f"Agent #{player_index + 1} gets {resources_str}.\n")

    # Prepare the answer list for doctest checking.
    status_str = "optimal" if prob.status == "optimal" else "non optimal"
    ans = [status_str, prob.value]
    for i in range(num_of_players):
        ans.append(utilities[i].value)
    return ans


def main():
    matrix_of_preferences = [[81, 19, 1],
                             [70, 1, 29]]
    egalitarian_allocation(matrix_of_preferences)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    main()
