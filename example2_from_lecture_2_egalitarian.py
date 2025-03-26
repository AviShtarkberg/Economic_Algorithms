import cvxpy

xw, xo = cvxpy.Variable(2)

utility_ami  = xw*100 + xo*20
utility_tami = (1-xw)*20 + (1-xo)*100

print("\nEgalitarian division")
min_utility = cvxpy.Variable()
prob = cvxpy.Problem(
    cvxpy.Maximize(min_utility),
    constraints = [0 <= xw, xw <= 1, 0 <= xo, xo <= 1,
                   min_utility<=utility_ami, min_utility<=utility_tami])
prob.solve()
print("status:", prob.status)
print("optimal value: ", prob.value)
print("Fractions given to Ami: ", xw.value, xo.value)
print("Utility of Ami", utility_ami.value)
print("Utility of Tami", utility_tami.value)