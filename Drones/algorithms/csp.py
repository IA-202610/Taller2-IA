from __future__ import annotations

from typing import TYPE_CHECKING
import time

if TYPE_CHECKING:
    from algorithms.problems_csp import DroneAssignmentCSP


def _init_stats(csp: "DroneAssignmentCSP", algorithm: str) -> None:
    csp.stats = {
        "algorithm": algorithm,
        "assignments_tried": 0,
        "consistency_checks": 0,
        "backtracks": 0,
        "domain_prunings": 0,
        "fc_wipeouts": 0,
        "ac3_revisions": 0,
        "ac3_arcs_processed": 0,
        "ac3_wipeouts": 0,
        "mrv_calls": 0,
        "degree_tiebreaks": 0,
        "lcv_calls": 0,
        "solved": False,
        "elapsed_time": 0.0,
    }


def backtracking_search(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    _init_stats(csp, "backtracking_search")
    start = time.perf_counter()

    def is_consistent(var: str, value: str, assignment: dict[str, str]) -> bool:
        csp.stats["consistency_checks"] += 1
        return csp.is_consistent(var, value, assignment)

    def backtrack(assignment: dict[str, str]):
        if csp.is_complete(assignment):
            return assignment

        var = csp.get_unassigned_variables(assignment)[0]

        for value in csp.domains[var]:
            csp.stats["assignments_tried"] += 1

            if is_consistent(var, value, assignment):
                csp.assign(var, value, assignment)
                result = backtrack(assignment)

                if result is not None:
                    return result

                csp.unassign(var, assignment)

        csp.stats["backtracks"] += 1
        return None

    result = backtrack({})
    csp.stats["solved"] = result is not None
    csp.stats["elapsed_time"] = time.perf_counter() - start
    return result


def backtracking_fc(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    _init_stats(csp, "backtracking_fc")
    start = time.perf_counter()

    def is_consistent(var: str, value: str, assignment: dict[str, str]) -> bool:
        csp.stats["consistency_checks"] += 1
        return csp.is_consistent(var, value, assignment)

    def forward_check(var: str, assignment: dict[str, str]):
        removed: dict[str, list[str]] = {}

        for neighbor in csp.get_neighbors(var):
            if neighbor not in assignment:
                removed[neighbor] = []

                for value in csp.domains[neighbor][:]:
                    if not is_consistent(neighbor, value, assignment):
                        csp.domains[neighbor].remove(value)
                        removed[neighbor].append(value)
                        csp.stats["domain_prunings"] += 1

                if len(csp.domains[neighbor]) == 0:
                    csp.stats["fc_wipeouts"] += 1
                    return False, removed

        return True, removed

    def restore_domains(saved_domains: dict[str, list[str]]) -> None:
        for var in saved_domains:
            csp.domains[var] = saved_domains[var][:]

    def backtrack(assignment: dict[str, str]):
        if csp.is_complete(assignment):
            return assignment

        var = csp.get_unassigned_variables(assignment)[0]

        for value in csp.domains[var][:]:
            csp.stats["assignments_tried"] += 1

            if is_consistent(var, value, assignment):
                saved_domains = {v: csp.domains[v][:] for v in csp.domains}

                csp.assign(var, value, assignment)
                ok, _removed = forward_check(var, assignment)

                if ok:
                    result = backtrack(assignment)
                    if result is not None:
                        return result

                restore_domains(saved_domains)
                csp.unassign(var, assignment)

        csp.stats["backtracks"] += 1
        return None

    result = backtrack({})
    csp.stats["solved"] = result is not None
    csp.stats["elapsed_time"] = time.perf_counter() - start
    return result


def backtracking_ac3(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    _init_stats(csp, "backtracking_ac3")
    start = time.perf_counter()
    from collections import deque

    def is_consistent(var: str, value: str, assignment: dict[str, str]) -> bool:
        csp.stats["consistency_checks"] += 1
        return csp.is_consistent(var, value, assignment)

    def values_compatible(Xi: str, vi: str, Xj: str, vj: str) -> bool:
        partial_assignment = {Xi: vi}
        return is_consistent(Xj, vj, partial_assignment)

    def revise(Xi: str, Xj: str) -> bool:
        revised = False
        csp.stats["ac3_revisions"] += 1

        for vi in csp.domains[Xi][:]:
            if not any(values_compatible(Xi, vi, Xj, vj) for vj in csp.domains[Xj]):
                csp.domains[Xi].remove(vi)
                csp.stats["domain_prunings"] += 1
                revised = True

        return revised

    def ac3(queue: deque[tuple[str, str]]) -> bool:
        while queue:
            Xi, Xj = queue.popleft()
            csp.stats["ac3_arcs_processed"] += 1

            if revise(Xi, Xj):
                if len(csp.domains[Xi]) == 0:
                    csp.stats["ac3_wipeouts"] += 1
                    return False

                for Xk in csp.get_neighbors(Xi):
                    if Xk != Xj:
                        queue.append((Xk, Xi))

        return True

    def backtrack(assignment: dict[str, str]):
        if csp.is_complete(assignment):
            return assignment

        var = csp.get_unassigned_variables(assignment)[0]

        for value in csp.domains[var][:]:
            csp.stats["assignments_tried"] += 1

            if is_consistent(var, value, assignment):
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

        csp.stats["backtracks"] += 1
        return None

    initial_queue = deque()
    for Xi in csp.domains:
        for Xj in csp.get_neighbors(Xi):
            initial_queue.append((Xi, Xj))

    if not ac3(initial_queue):
        csp.stats["solved"] = False
        csp.stats["elapsed_time"] = time.perf_counter() - start
        return None

    result = backtrack({})
    csp.stats["solved"] = result is not None
    csp.stats["elapsed_time"] = time.perf_counter() - start
    return result


def backtracking_mrv_lcv(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    _init_stats(csp, "backtracking_mrv_lcv")
    start = time.perf_counter()

    def is_consistent(var: str, value: str, assignment: dict[str, str]) -> bool:
        csp.stats["consistency_checks"] += 1
        return csp.is_consistent(var, value, assignment)

    def select_unassigned_variable(assignment: dict[str, str]) -> str:
        csp.stats["mrv_calls"] += 1
        unassigned = csp.get_unassigned_variables(assignment)

        mrv = min(unassigned, key=lambda var: len(csp.domains[var]))
        min_size = len(csp.domains[mrv])
        candidates = [v for v in unassigned if len(csp.domains[v]) == min_size]

        if len(candidates) > 1:
            csp.stats["degree_tiebreaks"] += 1
            return max(
                candidates,
                key=lambda var: len(
                    [n for n in csp.get_neighbors(var) if n not in assignment]
                ),
            )

        return mrv

    def order_domain_values(var: str, assignment: dict[str, str]) -> list[str]:
        csp.stats["lcv_calls"] += 1
        return sorted(
            csp.domains[var],
            key=lambda value: csp.get_num_conflicts(var, value, assignment),
        )

    def forward_check(var: str, assignment: dict[str, str]):
        removed: dict[str, list[str]] = {}

        for neighbor in csp.get_neighbors(var):
            if neighbor not in assignment:
                removed[neighbor] = []

                for value in csp.domains[neighbor][:]:
                    if not is_consistent(neighbor, value, assignment):
                        csp.domains[neighbor].remove(value)
                        removed[neighbor].append(value)
                        csp.stats["domain_prunings"] += 1

                if len(csp.domains[neighbor]) == 0:
                    csp.stats["fc_wipeouts"] += 1
                    return False, removed

        return True, removed

    def restore_domains(saved_domains: dict[str, list[str]]) -> None:
        for var in saved_domains:
            csp.domains[var] = saved_domains[var][:]

    def backtrack(assignment: dict[str, str]):
        if csp.is_complete(assignment):
            return assignment

        var = select_unassigned_variable(assignment)

        for value in order_domain_values(var, assignment):
            csp.stats["assignments_tried"] += 1

            if is_consistent(var, value, assignment):
                saved_domains = {v: csp.domains[v][:] for v in csp.domains}

                csp.assign(var, value, assignment)
                ok, _removed = forward_check(var, assignment)

                if ok:
                    result = backtrack(assignment)
                    if result is not None:
                        return result

                restore_domains(saved_domains)
                csp.unassign(var, assignment)

        csp.stats["backtracks"] += 1
        return None

    result = backtrack({})
    csp.stats["solved"] = result is not None
    csp.stats["elapsed_time"] = time.perf_counter() - start
    return result
