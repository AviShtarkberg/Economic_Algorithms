import cvxpy

xw, xo, xs = cvxpy.Variable(3)


utility_ami = 80*xw + 19*xo + 1*xs
utility_tami = 79*(1-xw) + 1*(1-xo) + 20*(1-xs)

constraints = [xw>=0, xw<=1, xo>=0, xo<=1, xs>=0, xs<=1]

problem = cvxpy.Problem(cvxpy.Maximize(utility_ami + utility_tami), constraints)

problem.solve()
print(xw.value)
print(xo.value)
print(xs.value)
print(problem.status)
