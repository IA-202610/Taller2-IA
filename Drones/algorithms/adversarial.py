from __future__ import annotations

import random
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

import algorithms.evaluation as evaluation
from world.game import Agent, Directions

if TYPE_CHECKING:
    from world.game_state import GameState


class MultiAgentSearchAgent(Agent, ABC):
    """
    Base class for multi-agent search agents (Minimax, AlphaBeta, Expectimax).
    """

    def __init__(self, depth: str = "2", _index: int = 0, prob: str = "0.0") -> None:
        self.index = 0  # Drone is always agent 0
        self.depth = int(depth)
        self.prob = float(
            prob
        )  # Probability that each hunter acts randomly (0=greedy, 1=random)
        self.evaluation_function = evaluation.evaluation_function

    @abstractmethod
    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone from the current GameState.
        """
        pass


class RandomAgent(MultiAgentSearchAgent):
    """
    Agent that chooses a legal action uniformly at random.
    """

    def get_action(self, state: GameState) -> Directions | None:
        """
        Get a random legal action for the drone.
        """
        legal_actions = state.get_legal_actions(self.index)
        return random.choice(legal_actions) if legal_actions else None


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Minimax agent for the drone (MAX) vs hunters (MIN) game.
    """

    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone using minimax.

        Tips:
        - The game tree alternates: drone (MAX) -> hunter1 (MIN) -> hunter2 (MIN) -> ... -> drone (MAX) -> ...
        - Use self.depth to control the search depth. depth=1 means the drone moves once and each hunter moves once.
        - Use state.get_legal_actions(agent_index) to get legal actions for a specific agent.
        - Use state.generate_successor(agent_index, action) to get the successor state after an action.
        - Use state.is_win() and state.is_lose() to check terminal states.
        - Use state.get_num_agents() to get the total number of agents.
        - Use self.evaluation_function(state) to evaluate leaf/terminal states.
        - The next agent is (agent_index + 1) % num_agents. Depth decreases after all agents have moved (full ply).
        - Return the ACTION (not the value) that maximizes the minimax value for the drone.
        """
        def minimax(state, depth, agent_index):
            #caso base
            if state.is_win() or state.is_lose() or depth == 0:
                return self.evaluation_function(state)
            
            num_agents = state.get_num_agents()
            next_agent = (agent_index + 1) % num_agents
            next_depth = depth - 1 if next_agent == 0 else depth

            #MAX 
            if agent_index == 0:
                value = -float("inf")
                for action in state.get_legal_actions(agent_index):
                    succesor = state.generate_successor(agent_index, action)
                    value = max(value, minimax(succesor, next_depth, next_agent))
                return value
            
            #MIN
            else:
                value = float("inf")
                for action in state.get_legal_actions(agent_index):
                    succesor = state.generate_successor(agent_index, action)
                    value = min(value, minimax(succesor, next_depth, next_agent))
                return value
        
        #Mejor accion
        legal_actions = state.get_legal_actions(0)
        best_value = -float("inf")
        best_actions = []

        for action in legal_actions:
            successor = state.generate_successor(0, action)
            # Llamamos al minimax para el primer Hunter (agente 1)
            value = minimax(successor, self.depth, 1)

            if value > best_value:
                best_value = value
                best_actions = [action]
            elif value == best_value:
                best_actions.append(action)

        import random
        chosen = random.choice(best_actions)
        
        successor = state.generate_successor(0, chosen)
        new_pos = successor.get_drone_position()
        from algorithms.evaluation import _revisitas
        _revisitas[new_pos] = _revisitas.get(new_pos, 0) + 1

        return chosen


class AlphaBetaAgent(MultiAgentSearchAgent):

    def get_action(self, state: GameState) -> Directions | None:

        def alphabeta(state, depth, agent_index, alpha, beta):
            # Caso base
            if state.is_win() or state.is_lose() or depth == 0:
                return self.evaluation_function(state)

            num_agents = state.get_num_agents()
            next_agent = (agent_index + 1) % num_agents
            next_depth = depth - 1 if next_agent == 0 else depth

            # MAX (dron)
            if agent_index == 0:
                value = -float("inf")
                for action in state.get_legal_actions(agent_index):
                    successor = state.generate_successor(agent_index, action)
                    value = max(value, alphabeta(successor, next_depth, next_agent, alpha, beta))
                    if value > beta:          # Poda: MAX no necesita explorar más
                        return value
                    alpha = max(alpha, value) # Actualizamos la mejor garantía de MAX
                return value

            # MIN (hunters)
            else:
                value = float("inf")
                for action in state.get_legal_actions(agent_index):
                    successor = state.generate_successor(agent_index, action)
                    value = min(value, alphabeta(successor, next_depth, next_agent, alpha, beta))
                    if value < alpha:         # Poda: MIN no necesita explorar más
                        return value
                    beta = min(beta, value)   # Actualizamos la mejor garantía de MIN
                return value

        # Mejor acción (igual que en Minimax)
        legal_actions = state.get_legal_actions(0)
        actions = [a for a in legal_actions if a != 'Stop']
        if not actions:
            actions = ['Stop']

        best_value = -float("inf")
        best_actions = []
        alpha = -float("inf")  # Alpha inicial
        beta = float("inf")    # Beta inicial

        for action in actions:
            successor = state.generate_successor(0, action)
            value = alphabeta(successor, self.depth, 1, alpha, beta)

            if value > best_value:
                best_value = value
                best_actions = [action]
            elif value == best_value:
                best_actions.append(action)

            alpha = max(alpha, best_value)  # Actualizamos alpha en la raíz también

        import random
        return random.choice(best_actions)


import random

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Expectimax agent with a mixed hunter model.
    Value = (1 - p) * min(child_values) + p * mean(child_values)
    """

    def get_action(self, state: GameState) -> Directions | None:
        
        def expectimax (state, depth, agent_index):
            #caso vase
            if state.is_win() or state.is_lose() or depth == 0:
                return self.evaluation_function(state)

            num_agents = state.get_num_agents()
            next_agent = (agent_index + 1) % num_agents
            next_depth = depth - 1 if next_agent == 0 else depth
            
            actions = state.get_legal_actions(agent_index)
            if not actions:
                return self.evaluation_function(state)

            # nodo max
            if agent_index == 0:
                v = -float("inf")
                for action in actions:
                    successor = state.generate_successor(agent_index, action)
                    v = max(v, expectimax(successor, next_depth, next_agent))
                return v

            # nodo chance -> hunters
            else:
                child_values = []
                for action in actions:
                    successor = state.generate_successor(agent_index, action)
                    child_values.append(expectimax(successor, next_depth, next_agent))

                min_val = min(child_values)
                mean_val = sum(child_values) / len(child_values)
                
                p = self.prob # Probabilidad de movimiento random
                return (1 - p) * min_val + p * mean_val

        # Mejor acción
        best_score = -float("inf")
        best_actions = []
        
        # Igual que en minimax se quita el stop para q no quede ahí
        legal_actions = state.get_legal_actions(0)
        drone_actions = [a for a in legal_actions if a != 'Stop']
        if not drone_actions: drone_actions = ['Stop']

        for action in drone_actions:
            successor = state.generate_successor(0, action)
            score = expectimax(successor, self.depth, 1)
            
            if score > best_score:
                best_score = score
                best_actions = [action]
            elif abs(score - best_score) < 0.001:
                best_actions.append(action)

        if not best_actions: return 'Stop'
        return random.choice(best_actions)
