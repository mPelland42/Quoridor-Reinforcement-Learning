
import heapq

#spaces are now point objects, moves are now actions
def AStar(game, startSpace, goalTest, heuristic):
    pq = [(0, startSpace, 0)]
    explored = set()
    while len(pq) > 0:
        currState = heapq.heappop(pq)
        explored.add(currState[1])
        #print "ASTAR", currState
        if goalTest(currState[1]):
            return currState
        else:
            for move in game.getPawnMoves(currState[1]):
                if move not in explored:
                    heapq.heappush(pq, (currState[2]+1 + heuristic(move), move, currState[2]+1))
    return (-1, -1, -1)
