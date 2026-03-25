from __future__ import annotations

from typing import TYPE_CHECKING
from algorithms import utils as u
import math


if TYPE_CHECKING:
    from world.game_state import GameState

_revisitas: dict[tuple[int, int], int] = {}


def evaluation_function(state: GameState) -> float:
    # Casos base (terminales)
    if state.is_lose():
        return -1000
    if state.is_win():
        return 1000

    score = state.get_score()
    dron = state.get_drone_position()
    layout = state.get_layout()
    pendientes = state.get_pending_deliveries()
    hunters = state.get_hunter_positions()

    # Valor base -> puntaje actual
    utilidad = float(score)

    # Si por alguna razón no hay pendientes y no entró en win, devolvemos utilidad actual
    if not pendientes:
        visitas = _revisitas.get(dron, 0)
        if visitas > 0:
            utilidad -= (visitas**2) * 10
        return float(utilidad)

    # Ubicar el objetivo más accesible con Dijstra
    # Osea, el delivery al que coste menos llegar
    menor_costo_d = float("inf")
    d_selec = None

    for delivery in pendientes:
        costo = u.bfs_distance(layout, dron, delivery)
        if costo < menor_costo_d:
            menor_costo_d = costo
            d_selec = delivery

    # penalización por distancia al delivery más cercano (d_selec)
    if not math.isinf(menor_costo_d):
        utilidad -= menor_costo_d

    # Analizar q tan seguro es el objetivo q seleccionó
    if d_selec:
        dist_hunter = float("inf")  # Cercanía del DRON al HUNTER
        dist_hunter_d = float("inf")  # Cercanía del HUNTER al DELIVERY <--- AMENAZA

        for hunter in hunters:
            # Del hunter al dron
            dr_h = u.bfs_distance(layout, dron, hunter)
            if dr_h < dist_hunter:
                dist_hunter = dr_h

            # Del hunter al delivery
            h_del = u.bfs_distance(layout, hunter, d_selec, hunter_restricted=True)
            if h_del < dist_hunter_d:
                dist_hunter_d = h_del

        # Acá se evalúa el riesgo según cercanía del hunter
        # PELIGRO EXTREMO!!! el hunter está a 1 paso del dron
        if dist_hunter <= 1:
            utilidad -= 1000.0
        # Si está a 4 o más pasos da más o menos igual así que se le resta mucho más poco
        elif dist_hunter >= 4 and not math.isinf(dist_hunter):
            utilidad -= 100.0 / dist_hunter

        # Se incentiva que el dron llegue al paquete antes que el cazador más cercano al d_selec
        if not math.isinf(menor_costo_d) and not math.isinf(dist_hunter_d):
            if menor_costo_d < dist_hunter_d:
                utilidad += 200.0

        # Castigo por deliveries pendientes
        faltan = len(state.get_pending_deliveries())
        utilidad -= faltan * 60

        # por si tiene el último deliver al PUTO LADO
        if faltan == 1 and menor_costo_d <= 1:
            utilidad += 1000

        # Revisitar la misma putisima celda
        visitas = _revisitas.get(dron, 0)
        if visitas > 0:
            utilidad -= (visitas**2) * 10

    return float(utilidad)
