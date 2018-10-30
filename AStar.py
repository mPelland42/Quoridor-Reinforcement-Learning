
from Game import Qoridor
import heapq


def AStar(game, startSpace, goalTest, heuristic):
    pq = [(0, startSpace, 0)]
    while len(pq) > 0:
        currState = heapq.heappop(pq)
        if goalTest(currState):
            return(currState)
        else:
            for move in game.getPawnMoves(currState[1]):
                heapq.heappush(pq, (currState[2]+1 + heuristic(move), currState[2]+1))
    return (-1, -1, -1)