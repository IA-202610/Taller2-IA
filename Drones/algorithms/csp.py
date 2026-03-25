from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from algorithms.problems_csp import DroneAssignmentCSP


def backtracking_search(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Basic backtracking search without optimizations.

    Tips:
    - An assignment is a dictionary mapping variables to values (e.g. {X1: Cell(1,2), X2: Cell(3,4)}).
    - Use csp.assign(var, value, assignment) to assign a value to a variable.
    - Use csp.unassign(var, assignment) to unassign a variable.
    - Use csp.is_consistent(var, value, assignment) to check if an assignment is consistent with the constraints.
    - Use csp.is_complete(assignment) to check if the assignment is complete (all variables assigned).
    - Use csp.get_unassigned_variables(assignment) to get a list of unassigned variables.
    - Use csp.domains[var] to get the list of possible values for a variable.
    - Use csp.get_neighbors(var) to get the list of variables that share a constraint with var.
    - Add logs to measure how good your implementation is (e.g. number of assignments, backtracks).

    You can find inspiration in the textbook's pseudocode:
    Artificial Intelligence: A Modern Approach (4th Edition) by Russell and Norvig, Chapter 5: Constraint Satisfaction Problems
    """
    def backtrack(assignment: dict[str, str]):
        if csp.is_complete(assignment):
            return assignment
        var = csp.get_unassigned_variables(assignment)[0]
        for value in csp.domains[var]:
            if csp.is_consistent(var, value, assignment):
                csp.assign(var, value, assignment)
                result = backtrack(assignment)
                if result is not None:
                    return result
                csp.unassign(var, assignment)
        return None
    return backtrack({})


def backtracking_fc(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Backtracking search with Forward Checking.

    Tips:
    - Forward checking: After assigning a value to a variable, eliminate inconsistent values from
      the domains of unassigned neighbors. If any neighbor's domain becomes empty, backtrack immediately.
    - Save domains before forward checking so you can restore them on backtrack.
    - Use csp.get_neighbors(var) to get variables that share constraints with var.
    - Use csp.is_consistent(neighbor, val, assignment) to check if a value is still consistent.
    - Forward checking reduces the search space by detecting failures earlier than basic backtracking.
    """
    def forward_check(var, assignment):
        removed = {}
        for neighbor in csp.get_neighbors(var):
            if neighbor not in assignment:
                removed[neighbor] = []
                for value in csp.domains[neighbor][:]:
                    if not csp.is_consistent(neighbor, value, assignment):
                        csp.domains[neighbor].remove(value)
                        removed[neighbor].append(value)
                if len(csp.domains[neighbor]) == 0:
                    return False, removed
        return True, removed
    def restore_domains(removed):
        for var in removed:
            csp.domains[var].extend(removed[var])
    def backtrack(assignment):
        if csp.is_complete(assignment):
            return assignment
        var = csp.get_unassigned_variables(assignment)[0]
        for value in csp.domains[var]:
            if csp.is_consistent(var, value, assignment):
                csp.assign(var, value, assignment)
                ok, removed = forward_check(var, assignment)
                if ok:
                    result = backtrack(assignment)
                    if result is not None:
                        return result
                restore_domains(removed)
                csp.unassign(var, assignment)
        return None
    return backtrack({})


def backtracking_ac3(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Backtracking search with AC-3 arc consistency.

    Tips:
    - AC-3 enforces arc consistency: for every pair of constrained variables (Xi, Xj), every value
      in Xi's domain must have at least one supporting value in Xj's domain.
    - Run AC-3 before starting backtracking to reduce domains globally.
    - After each assignment, run AC-3 on arcs involving the assigned variable's neighbors.
    - If AC-3 empties any domain, the current assignment is inconsistent - backtrack.
    - You can create helper functions such as:
      - a values_compatible function to check if two variable-value pairs are consistent with the constraints.
      - a revise function that removes unsupported values from one variable's domain.
      - an ac3 function that manages the queue of arcs to check and calls revise.
      - a backtrack function that integrates AC-3 into the search process.
    """
    from collections import deque
    import copy

    def values_compatible(Xi, vi, Xj, vj):
        assignment = {Xi: vi}
        return csp.is_consistent(Xj, vj, assignment)

    def revise(Xi, Xj):
        revised = False

        for vi in csp.domains[Xi][:]:
            if not any(values_compatible(Xi, vi, Xj, vj) for vj in csp.domains[Xj]):
                csp.domains[Xi].remove(vi)
                revised = True

        return revised

    def ac3(queue):
        while queue:
            Xi, Xj = queue.popleft()

            if revise(Xi, Xj):
                if len(csp.domains[Xi]) == 0:
                    return False

                for Xk in csp.get_neighbors(Xi):
                    if Xk != Xj:
                        queue.append((Xk, Xi))

        return True

    def backtrack(assignment):
        if csp.is_complete(assignment):
            return assignment

        var = csp.get_unassigned_variables(assignment)[0]

        for value in csp.domains[var][:]:
            if csp.is_consistent(var, value, assignment):

                saved_domains = {v: csp.domains[v][:] for v in csp.domains}

                csp.assign(var, value, assignment)

                queue = deque()
                for neighbor in csp.get_neighbors(var):
                    queue.append((neighbor, var))

                if ac3(queue):
                    result = backtrack(assignment)
                    if result is not None:
                        return result

                for v in csp.domains:
                    csp.domains[v] = saved_domains[v][:]

                csp.unassign(var, assignment)

        return None

    from collections import deque
    initial_queue = deque()

    for Xi in csp.domains:
        for Xj in csp.get_neighbors(Xi):
            initial_queue.append((Xi, Xj))

    if not ac3(initial_queue):
        return None

    return backtrack({})
    


def backtracking_mrv_lcv(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Backtracking with Forward Checking + MRV + LCV.

    Tips:
    - Combine the techniques from backtracking_fc, mrv_heuristic, and lcv_heuristic.
    - MRV (Minimum Remaining Values): Select the unassigned variable with the fewest legal values.
      Tie-break by degree: prefer the variable with the most unassigned neighbors.
    - LCV (Least Constraining Value): When ordering values for a variable, prefer
      values that rule out the fewest choices for neighboring variables.
    - Use csp.get_num_conflicts(var, value, assignment) to count how many values would be ruled out for neighbors if var=value is assigned.
    """
    # TODO: Implement your code here (BONUS)
    def select_unassigned_variable(assignment):
        unassigned = csp.get_unassigned_variables(assignment)

        mrv = min(unassigned, key=lambda var: len(csp.domains[var]))

        min_size = len(csp.domains[mrv])
        candidates = [v for v in unassigned if len(csp.domains[v]) == min_size]

        if len(candidates) > 1:
            return max(candidates, key=lambda var: len([
                n for n in csp.get_neighbors(var) if n not in assignment
            ]))

        return mrv

    def order_domain_values(var, assignment):
        return sorted(
            csp.domains[var],
            key=lambda value: csp.get_num_conflicts(var, value, assignment)
        )

    def forward_check(var, assignment):
        removed = {}

        for neighbor in csp.get_neighbors(var):
            if neighbor not in assignment:
                removed[neighbor] = []

                for value in csp.domains[neighbor][:]:
                    if not csp.is_consistent(neighbor, value, assignment):
                        csp.domains[neighbor].remove(value)
                        removed[neighbor].append(value)

                if len(csp.domains[neighbor]) == 0:
                    return False, removed

        return True, removed

    def restore_domains(removed):
        for var in removed:
            csp.domains[var].extend(removed[var])

    def backtrack(assignment):
        if csp.is_complete(assignment):
            return assignment

        var = select_unassigned_variable(assignment)

        for value in order_domain_values(var, assignment):
            if csp.is_consistent(var, value, assignment):

                csp.assign(var, value, assignment)

                ok, removed = forward_check(var, assignment)

                if ok:
                    result = backtrack(assignment)
                    if result is not None:
                        return result

                restore_domains(removed)
                csp.unassign(var, assignment)

        return None

    return backtrack({})
