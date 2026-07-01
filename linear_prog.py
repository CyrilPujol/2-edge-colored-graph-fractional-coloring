from scipy.optimize import linprog
import numpy as np
from fractions import Fraction
from math import lcm

def frac(x):
    return Fraction(x).limit_denominator()

def int_to_symbol(n):
    symbols = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789*&§$€%£<>æ®†Úπ‡∂ﬁ¬µ≈©◊ß~"
    if n < 0 or n >= len(symbols):
        raise ValueError("Index out of range - no more symbols available.")
    return symbols[n]

class LinearProgram:
    def __init__(self):
        self.objective = []  # Coefficients of the objective function
        self.constraints = []  # Coefficients of constraints
        self.bounds = []  # RHS of constraints
        self.ineq_types = []  # Type of inequality: <=, ==, or >=
        self.var_bounds = []  # Variable bounds
        self.num_vars = 0
        self.minimize = True  # Default to minimization
        self.result = None  # Stores the solution after solve() is called

    def add_variable(self, lower=0, upper=None):
        """Adds a variable with optional bounds."""
        self.var_bounds.append((lower, upper))
        self.num_vars += 1
        return self.num_vars - 1  # Return the index of the new variable
    
    def add_variables(self, k, lower=0, upper=None):
        """Adds k variables with optional bounds."""
        return [self.add_variable(lower, upper) for _ in range(k)]
    
    def add_variables_from(self, iter, lower=0, upper=None):
        """Adds k variables with optional bounds."""
        return {x : self.add_variable(lower, upper) for x in iter}

    def dict_to_list(self, coeff_dict):
        """Converts a dictionary {i: coeff} to a full list of coefficients."""
        coeff_list = [0] * self.num_vars
        for i, coeff in coeff_dict.items():
            if i >= self.num_vars:
                raise ValueError("Coefficient index out of range.")
            coeff_list[i] = coeff
        return coeff_list

    def set_objective(self, coefficients, minimize=True):
        """Sets the coefficients of the objective function and whether to minimize or maximize."""
        if isinstance(coefficients, dict):
            coefficients = self.dict_to_list(coefficients)
        if len(coefficients) != self.num_vars:
            raise ValueError("Objective function must match the number of variables.")
        
        self.objective = coefficients if minimize else [-c for c in coefficients]
        self.minimize = minimize

    def add_constraint(self, coefficients, bound, ineq_type='<='):
        """Adds a constraint of type <=, ==, or >=."""
        if isinstance(coefficients, dict):
            coefficients = self.dict_to_list(coefficients)
        if len(coefficients) != self.num_vars:
            raise ValueError("Constraint must match the number of variables.")
        self.constraints.append(coefficients)
        self.bounds.append(bound)
        self.ineq_types.append(ineq_type)

    def solve(self):
        """Solves the linear program."""
        if not self.objective:
            raise ValueError("Objective function not set.")
        
        A_ub, b_ub, A_eq, b_eq = [], [], [], []
        for coef, bound, ineq in zip(self.constraints, self.bounds, self.ineq_types):
            if ineq == '<=':
                A_ub.append(coef)
                b_ub.append(bound)
            elif ineq == '>=':
                A_ub.append([-c for c in coef])
                b_ub.append(-bound)
            elif ineq == '==':
                A_eq.append(coef)
                b_eq.append(bound)
            else:
                raise ValueError("Invalid constraint type. Use '<=', '>=', or '=='.")
        
        self.A_ub = A_ub
        self.b_ub = b_ub
        self.A_eq = A_eq
        self.b_eq = b_eq
        
        self.result = linprog(
            c=self.objective, A_ub=A_ub or None, b_ub=b_ub or None,
            A_eq=A_eq or None, b_eq=b_eq or None,
            bounds=self.var_bounds, method='highs'
        )
        return self.result

    def print_problem(self):
        """Prints the current linear program."""
        print("Objective:", self.objective, "(Minimize)" if self.minimize else "(Maximize)")
        print("Constraints:")
        for coef, bound, ineq in zip(self.constraints, self.bounds, self.ineq_types):
            print(f"  {coef} {ineq} {bound}")
        print("Variable Bounds:", self.var_bounds)
    
    def makeColouring(self, varSets):
        """Generates a coloring from the LP solution with consecutive integer labels."""
        if self.result is None or not self.result.success:
            raise ValueError("Cannot generate coloring: No valid solution found.")
        if len(varSets) != self.num_vars:
            raise ValueError("varSets size must match the number of variables.")
        
        coloring = {}
        color_map = {}  # Maps original color indices to consecutive integers
        next_label = 0
        
        for var_idx, weight in enumerate(self.result.x):
            if weight > 0:  # Ignore null weights
                if var_idx not in color_map:
                    color_map[var_idx] = next_label
                    next_label += 1
                new_label = color_map[var_idx]
                
                for vertex in varSets[var_idx]:
                    if vertex not in coloring:
                        coloring[vertex] = {}
                    coloring[vertex][new_label] = weight
        
        # Verify that the sum of weights on each vertex is 1
        for vertex, colors in coloring.items():
            total_weight = sum(colors.values())
            if not (np.isclose(total_weight, 1.0) or total_weight >= 1.0):
                raise ValueError(f"Weight sum error for vertex {vertex}: {total_weight}")
        
        return coloring
    
    def Denormalize(self, coloring):
        """Denormalizes a coloring by multiplying all weights by the common denominator."""
        denominators = [Fraction(weight).limit_denominator(10000).denominator for vertex in coloring for weight in coloring[vertex].values()]
        k = lcm(*denominators)
        
        denormalized_coloring = {
            vertex: {color: int(np.round(weight * k)) for color, weight in colors.items()}
            for vertex, colors in coloring.items()
        }

        return denormalized_coloring

    def print_coloring(self, coloring):
        """Prints the coloring in a formatted way."""
        coloring = self.Denormalize(coloring)
        dic_content = sorted(coloring.items(), key=lambda x: x[0])
        for key, val in dic_content:
            l_val = sorted(val.items(), key=lambda x: x[0])
            print(f"{key} : {''.join([f'{int_to_symbol(c)}' + '-' * (v - 1) for c, v in l_val])}")

    def dual_result(self):
        """Returns the dual solution using the primal result."""
        if self.result is None:
            raise ValueError("Solve the primal first to get dual result.")
        
        dual_vars = {}
        for i, marginal in enumerate(self.result.ineqlin.marginals):
            dual_vars[f"y{i}"] = marginal
        
        if self.result.eqlin.marginals.size > 0:
            for i, marginal in enumerate(self.result.eqlin.marginals):
                dual_vars[f"z{i}"] = marginal
        
        dual_opt = np.dot(self.b_ub, self.result.ineqlin.marginals)
        if self.b_eq:
            dual_opt += np.dot(self.b_eq, self.result.eqlin.marginals)
        
        return {"variables": dual_vars, "optimal_value": dual_opt}

    def print_result(self, VarNames=None, ShowNulVariables=False,convertToFrac=False, ShowVariables=True):
        """Prints the solution if available."""
        if self.result is None:
            print("No solution available. Please call solve() first.")
            return
        if not self.result.success:
            print("Optimization failed:", self.result.message)
            return
        
        if ShowVariables:
            print("Variable values:")
            for i, value in enumerate(self.result.x):
                if not ShowNulVariables and value == 0:
                    continue
                var_name = VarNames[i] if VarNames and i < len(VarNames) else f"x{i}"
                if convertToFrac :
                    print(f"  {var_name} = {frac(value)}")
                else:
                    print(f"  {var_name} = {np.round(value,5)}")
        
        opt_value = self.result.fun if self.minimize else -self.result.fun
        if convertToFrac :
            print(f"Optimal value: {opt_value} = {frac(opt_value)}")
        else:
            print(f"Optimal value: {opt_value}")

if __name__ == '__main__':
    print("__BEGIN__")

    lp = LinearProgram()
    
    singl = [{1},{2},{3},{4}]
    l_singl = lp.add_variables(len(singl)) #4 singletons

    pairs = [{1,2},{1,3},{1,4},{2,3},{2,4},{3,4}]
    l_pairs = lp.add_variables(len(pairs)) #6 edges
    
    lp.set_objective([1]*lp.num_vars)
    
    for u in range(1,5):
        lp.add_constraint({l_singl[i] : 1 if u in S else 0 for (i,S) in enumerate(singl) }|{l_pairs[i] : 1 if u in S else 0 for (i,S) in enumerate(pairs)}, 1, ">=")
        # sum x_S for u in S >=1
    
    a = 1/6.
    for u,v in [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)]:
        lp.add_constraint({l_singl[i] : 1 if u in S and v in S else 0 for (i,S) in enumerate(singl) }|{l_pairs[i] : 1 if u in S and v in S else 0 for (i,S) in enumerate(pairs)}, a, "<=")

    lp.print_problem()
    result = lp.solve()
    print("Primal variables:", np.round(result.x,3), "Primal opt:", np.round(result.fun,2))

    print(lp.dual_result())

    print("___END___")
