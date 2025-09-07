from abc import ABCMeta, abstractmethod
from state import State
import heapq
from collections import deque, defaultdict
from action import Action, ActionType
from itertools import product
from sortedcontainers import SortedList
import math
import random
import bisect




class Heuristic(metaclass=ABCMeta):
    def __init__(self, initial_state: 'State'):
        print('#Start preprocessing', flush=True)
        self.goal_locations = {}
        self.only_goals = set()
        self.all_distances = {}
        self.all_boxes = {}
        for row in range(len(State.goals)):
            for col in range(len(State.goals[row])):
                if State.goals[row][col] != '':
                    goal = State.goals[row][col]
                    self.goal_locations[(row,col)] = goal
                    self.only_goals.add(goal)
                if initial_state.boxes[row][col] != '':
                    self.all_boxes[(row,col)] = initial_state.boxes[row][col]
        self.all_distances = State.all_distances
        print('#Finish preprocessing', flush=True)



    def return_goals(self):
        return self.only_goals

    def h(self, state: 'State') -> 'int':
        result = []
        result.append(self.goals_not_achieved(state))

        return sum(result)
    
    def manhattan_distance_agent_box_only(self, state:'State'):
        result = 0
        for row in range(len(state.boxes)):
            for col in range(len(state.boxes[row])):
                if state.boxes[row][col] != '':
                    box = state.boxes[row][col]
                    agent, agent_row, agent_col, dist = self.find_agent_based_on_box(state, box, row,col)
                    result += abs(agent_row - row) + abs(agent_col - col)
        return result

    def manhattan_distance(self, state: 'State') -> 'int': 
        result = 0
        for (row,col),goal in self.goal_locations.items():
            if '0' <= goal <= '9':
                agent = ord(goal) - ord('0')
                result += abs(state.agent_rows[agent] - row) + abs(state.agent_cols[agent] - col)
            elif 'A' <= goal <= 'Z':
                closest_box = None
                min_dist = float('inf')
                
                for row2 in range(len(state.boxes)):
                    for col2 in range(len(state.boxes[row])):
                        if state.boxes[row2][col2] == goal:
                            dist = abs(row2 - row) + abs(col2 - col)
                            if dist < min_dist:
                                min_dist = dist
                                closest_box = [row2,col2]
                if closest_box:
                    result += min_dist
        return result
        
    def manhattan_distance_with_agent(self, state: 'State') -> 'int':  
        result = 0
        for (row,col),goal in self.goal_locations.items():
            if '0' <= goal <= '9':
                agent = ord(goal) - ord('0')
                result += abs(state.agent_rows[agent] - row) + abs(state.agent_cols[agent] - col)
            elif 'A' <= goal <= 'Z':
                closest_box = None
                min_dist = float('inf')     
                for row2 in range(len(state.boxes)):
                    for col2 in range(len(state.boxes[row])):
                        if state.boxes[row2][col2] == goal:
                            dist = abs(row2 - row) + abs(col2 - col)
                            if dist < min_dist:
                                min_dist = dist
                                closest_box = [row2,col2]
                if closest_box:
                    result += min_dist

                if state.boxes[row][col] == '':
                    color = State.box_colors[ord(goal) - ord('A')]
                    min_dist_box_agent = float('inf')
                    closest_agent = None
                    for agent in range(len(State.agent_colors)): # include if 2 agents have the same color
                        if State.agent_colors[agent] == color:
                            dist = abs(state.agent_rows[agent] - closest_box[0]) + abs(state.agent_cols[agent] - closest_box[1])
                            if dist < min_dist_box_agent:
                                min_dist_box_agent = dist
                                closest_agent = [state.agent_rows[agent],state.agent_cols[agent]]
                    
                    if closest_agent:
                        result += abs(closest_agent[0] - closest_box[0]) + abs(closest_agent[1] - closest_box[1])
        
        return result


    def penalization_based_on_goals(self, state: 'State'):
        result = 0
        only_goals = [(g[2],g[3],g[4]) for g in self.ordered_goals]
        agent_goal_only_intheend = True

        for (row,col),goal in self.goal_locations.items():
            if '0' <= goal <= '9':
                agent, arow, acol, dist = self.find_agent_based_on_goal(state, goal, row, col)
                penalty = 1
                pos_in_priority = (len(self.ordered_goals) - only_goals.index((goal,row,col)))+1
                if agent_goal_only_intheend and self.agent_boxes_all_in_goals(state, agent, row, col):                
                    for (walls2, near_goals, goal2, row2, col2) in self.ordered_goals:
                        if (row,col) == (row2,col2) and walls2 >= self.max_walls_not_achieved_in_goal(state):
                            if (state.agent_rows[int(agent)],state.agent_cols[int(agent)]) != (row,col) and self.goal_blocked(state, goal,row,col):
                                penalty = math.log(walls2)
                            break
                    result += dist * (pos_in_priority**2) * penalty

            elif 'A' <= goal <= 'Z':
                penalty = 1
                closest_box = self.find_box_based_on_goal(state, goal,row,col)
                pos_in_priority = (len(self.ordered_goals) - only_goals.index((goal,row,col)))+1
                dist = closest_box[3] 
                for (walls2, near_goals, goal2, row2, col2) in self.ordered_goals:
                    if (row,col) == (row2,col2):
                        if state.joint_action != None and any(action.type in [ActionType.Push, ActionType.Pull] for action in state.joint_action):
                            break
                        if walls2 == self.max_walls_not_achieved_in_goal(state) and state.boxes[row][col] == '' and self.goal_blocked(state,goal,row,col):
                            if walls2+near_goals == 0:
                                continue
                            penalty = math.log(walls2+near_goals)
                            break
                result += dist * (pos_in_priority**2) * penalty
        return result

    def manhattan_distance_avoiding_walls(self, state: 'State') -> 'int':
        result = 0
        distance_agent_box_reduced_by = 0
        agent_goal_only_in_the_end = True
        for (row,col),goal in self.goal_locations.items():
            if '0' <= goal <= '9':
                agent = ord(goal) - ord('0')
                distance_map = self.all_distances[(row,col)]
                agent_row, agent_col = state.agent_rows[agent], state.agent_cols[agent]
                dist = distance_map[agent_row][agent_col]
                if agent_goal_only_in_the_end and self.agent_boxes_all_in_goals(state, agent, row, col):
                    result += dist
            elif 'A' <= goal <= 'Z':
                distance_map = self.all_distances[(row,col)]
                closest_box = self.find_box_based_on_goal(state, goal, row, col)
                box, box_row, box_col = closest_box[0], closest_box[1], closest_box[2]
                dist = distance_map[box_row][box_col]
                result += dist

        if distance_agent_box_reduced_by > 0:
            for agent in range(len(state.agent_rows)):
                agent, agent_row, agent_col = str(agent), state.agent_rows[agent], state.agent_cols[agent]
                box_info = self.find_box_based_on_agent(state, agent, agent_row, agent_col)
                if box_info is not None:
                    box, box_row, box_col,distance = box_info
                    result += distance/distance_agent_box_reduced_by
        return result
    
    def return_1(self, state):
        return 1

    def find_goals_near_goal(self, state, goal, row, col):
        goals = {}
        for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            if State.goals[row+dx][col+dy] != '':
                goals[State.goals[row+dx][col+dy]] = [row+dx,col+dy]
        return goals

    def goal_blocked(self, state, goal, grow, gcol):
        near_goals = self.find_goals_near_goal(state, goal, grow,gcol)
        ordered_goals = [(g[2],g[3],g[4]) for g in self.ordered_goals]
        goal_index = ordered_goals.index((goal,grow,gcol))
        walls_near = self.ordered_goals[goal_index][0]
        # if walls_near + len(near_goals) < 8:
        #     return False
        for g,[r,c] in near_goals.items():
            gindex = ordered_goals.index((g,r,c))
            if state.boxes[r][c] != g and goal_index < gindex:
                return False
        return True

    def is_respect_priority(self, state, goal, grow,gcol):
        ordered_goals = [(g[2],g[3],g[4]) for g in self.ordered_goals]
        goal_index = ordered_goals.index((goal,grow,gcol))
        for i,(_,_,g,r,c) in enumerate(self.ordered_goals):
            if i > goal_index:
                if '0' <= g <= '9' and state.agent_rows[ord(str(g))-ord('0')]==r and state.agent_cols[ord(str(g))-ord('0')]==c: 
                    return False
                if 'A' <= g <= 'Z' and state.boxes[r][c] == g:
                    return False
        return True

    def precompute_distance_map(self, state, goal, goal_row, goal_col):
        max_rows = len(state.boxes)
        max_cols = len(state.boxes[0])
        distance_map = [[float('inf')]*max_cols for _ in range(max_rows)]
        queue = deque([(goal_row, goal_col, 0)])
        while queue:
            row,col,dist = queue.popleft()
            if distance_map[row][col] != float('inf'):
                continue
            distance_map[row][col] = dist
            for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                neighbor = (row + dx, col + dy)
                if 0 <= neighbor[0] < max_rows and 0 <= neighbor[1] < max_cols and not State.walls[neighbor[0]][neighbor[1]]:
                    queue.append((neighbor[0],neighbor[1], dist+1))
        return distance_map

    def precompute_distance_map_with_goals_as_walls(self, state, goal, goal_row, goal_col):
        max_rows = len(state.boxes)
        max_cols = len(state.boxes[0])
        distance_map = [[float('inf')]*max_cols for _ in range(max_rows)]
        queue = deque([(goal_row, goal_col, 0)])
        while queue:
            row,col,dist = queue.popleft()
            if distance_map[row][col] != float('inf'):
                continue
            distance_map[row][col] = dist
            for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                neighbor = (row + dx, col + dy)
                if 0 <= neighbor[0] < max_rows and 0 <= neighbor[1] < max_cols:
                    if not State.walls[neighbor[0]][neighbor[1]] and (State.goals[row][col] == '' or state.boxes[row][col] == '' or state.boxes[neighbor[0]][neighbor[1]] != State.goals[neighbor[0]][neighbor[1]]):
                        queue.append((neighbor[0],neighbor[1], dist+1))
        return distance_map

    def ordered_goals_based_on_difficulty(self,state: 'State'):
        queue = []
        for (row,col),goal in self.goal_locations.items():
            walls_near = 0
            goals_near = 0
            free_spaces = 0
            # for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            for dx in [-1,0,1]:
                for dy in [-1,0,1]:
                    if dx == 0 and dy == 0:
                        continue
                    if State.walls[row+dx][col+dy]:
                        walls_near += 1
                    if State.goals[row+dx][col+dy] != '':
                        goals_near += 1
            heapq.heappush(queue, (walls_near, goals_near, goal, row, col))
        return queue
    
    def goals_not_achieved(self, state: 'State'):
        result = len(self.goal_locations)
        for (row,col),goal in self.goal_locations.items():
            if '0' <= goal <= '9' and state.agent_rows[ord(goal)-ord('0')] == row and state.agent_cols[ord(goal)-ord('0')] == col:
                result -= 1
            elif 'A' <= goal <= 'Z' and state.boxes[row][col] == goal:
                result -= 1
        return result
    
    def goals_achieved(self, state: 'State'):
        result = 0
        for (row,col),goal in self.goal_locations.items():
            if '0' <= goal <= '9' and state.agent_rows[ord(goal)-ord('0')] == row and state.agent_cols[ord(goal)-ord('0')] == col:
                result += 1
            elif 'A' <= goal <= 'Z' and state.boxes[row][col] == goal:
                result += 1
        return result
    
    def goals_achieved_per_agent(self, state):
        results = {}
        for agent in range(len(state.agent_rows)):
            result = 0
            agent_color = State.agent_colors[agent]
            for (row,col),goal in self.goal_locations.items():
                if '0' <= goal <= '9' and goal == str(agent) and state.agent_rows[ord(goal)-ord('0')] == row and state.agent_cols[ord(goal)-ord('0')] == col:
                    result += 1
                elif 'A' <= goal <= 'Z' and State.box_colors[ord(goal)-ord('A')]==agent_color and state.boxes[row][col] == goal:
                    result += 1  
            results[agent] = result
        return results

    def which_goals_are_achieved(self, state: 'State'):
        goals_achieved = []
        for row in range(len(State.goals)):
            for col in range(len(State.goals[row])):
                if State.goals[row][col] != '':
                    goal = State.goals[row][col]
                    if 'A' <= goal <= 'Z' and state.boxes[row][col] == goal:
                        goals_achieved.append((goal,row,col))
                    elif '0' <= goal <= '9' and state.agent_rows[ord(goal)-ord('0')] == row and state.agent_cols[ord(goal)-ord('0')] == col:
                        goals_achieved.append((goal,row,col))
        return goals_achieved

    def find_box_based_on_goal(self, state : 'State', goal, grow, gcol):
        boxes = []
        distance_map = self.all_distances[(grow,gcol)]
        for (row,col,box) in state.boxes_locations:
                if box==goal:
                    boxes.append((state.boxes[row][col],row,col, distance_map[row][col])) 
        return sorted(boxes, key=lambda x: x[3])[0]
    
    def find_agent_based_on_box(self, state: 'State', box, row, col):
        agents = []
        for ag in range(len(state.agent_rows)):
            agent_color = State.agent_colors[ag]
            box_color = State.box_colors[ord(box) - ord('A')]
            if agent_color == box_color:
                agent_row, agent_col = state.agent_rows[ag], state.agent_cols[ag]
                dist = math.sqrt((agent_row - row) ** 2 + (agent_col - col) ** 2)
                agents.append((ag, agent_row, agent_col,dist))
        return sorted(agents, key=lambda x:x[3])[0]
    
    def find_box_based_on_agent(self, state,agent, arow, acol):
        boxes = []
        agent_color = State.agent_colors[ord(str(agent))-ord('0')]
        for row in range(len(state.boxes)):
            for col in range(len(state.boxes[row])):
                box = state.boxes[row][col]
                if box != '' and State.box_colors[ord(box)-ord('A')] == agent_color and State.goals[row][col] != box:
                    dist = self.all_distances[(arow,acol)][row][col]
                    boxes.append([box, row, col, dist])
        if boxes == []:
            return None
        return sorted(boxes, key=lambda x:x[3])[0]
    
    def find_agent_based_on_goal(self, state, goal, grow, gcol):
        agent = ord(str(goal)) - ord('0')
        agent_row, agent_col = state.agent_rows[agent], state.agent_cols[agent],
        return (agent, agent_row, agent_col, self.all_distances[(grow,gcol)][agent_row][agent_col])

    def agent_boxes_all_in_goals(self, state, agent, arow, acol):
        agent_color = State.agent_colors[ord(str(agent))-ord('0')]
        for row in range(len(state.boxes)):
            for col in range(len(state.boxes[row])):
                if state.boxes[row][col]!='':
                    box = state.boxes[row][col]
                    if State.box_colors[ord(box)-ord('A')] == agent_color:
                        if State.goals[row][col] == '' or self.goal_locations[(row,col)] != box:
                            return False
        return True

    def max_walls_not_achieved_in_goal(self, state):
        all_walls = []
        for (walls, _, goal, row, col) in self.ordered_goals:
            if state.boxes[row][col] != goal:
                all_walls.append(walls)
        if all_walls==[]:
            return 1
        return max(all_walls)

    def distance_agent_goal(self,state,agent,arow,acol):
        for (row,col),goal in self.goal_locations.items():
            if goal == agent:
                return self.all_distances[(arow,acol)][row][col]

    def get_g(self, state):
        return state.g
    
    def get_new_atoms(self,state):
        return state.new_atoms

    def agent_is_moving_a_box(self,state):
        if state.joint_action == None:
            return True
        if any(action.type in [ActionType.Push, ActionType.Pull] for action in state.joint_action):
            return True
        return False

    def only_manhattan_distance_avoiding_walls(self, state: 'State') -> 'int':
        result = 0
        agent_goal_only_intheend = True

        for (row,col),goal in self.goal_locations.items():
            if '0' <= goal <= '9':
                agent = ord(goal) - ord('0')
                distance_map = self.all_distances[(row,col)]
                agent_row, agent_col = state.agent_rows[agent], state.agent_cols[agent]
                dist = distance_map[agent_row][agent_col]
                if agent_goal_only_intheend and self.agent_boxes_all_in_goals(state, agent, row, col):
                    result += dist
            elif 'A' <= goal <= 'Z':
                distance_map = self.all_distances[(row,col)]
                closest_box = self.find_box_based_on_goal(state, goal, row, col)
                box, box_row, box_col = closest_box[0], closest_box[1], closest_box[2]
                dist = distance_map[box_row][box_col]
                result += dist
        return result

    def get_new_goals_atoms(self, state, goals_seen):
        if state.parent == None:
            return 0
        for atom in state.atoms:
            if atom[0] == 'goal' and self.goals_achieved(state) == self.goals_achieved(state.parent):
                if atom not in goals_seen:#state.parent.atoms:
                    # print('aaa',atom, flush=True)  
                    return 0
                # s = state.parent
                # while s != None:
                    # if atom in s.atoms:
                        # return 1
                    # s = s.parent
                # print('aaa',atom, flush=True)
                # return 1
        return 1


    @abstractmethod
    def f(self, state: 'State') -> 'int': pass
    
    @abstractmethod
    def __repr__(self): raise NotImplementedError

class HeuristicAStar(Heuristic):
    def __init__(self, initial_state: 'State'):
        super().__init__(initial_state)
    
    def f(self, state: 'State') -> 'int':
        return state.g + self.h(state)
    
    def __repr__(self):
        return 'A* evaluation'

class HeuristicWeightedAStar(Heuristic):
    def __init__(self, initial_state: 'State', w: 'int'):
        super().__init__(initial_state)
        self.w = w
    
    def f(self, state: 'State') -> 'int':
        return state.g + self.w * self.h(state)
    
    def __repr__(self):
        return 'WA*({}) evaluation'.format(self.w)

class HeuristicGreedy(Heuristic):
    def __init__(self, initial_state: 'State'):
        super().__init__(initial_state)
    
    def f(self, state: 'State') -> 'int':
        return self.h(state)
    
    def __repr__(self):
        return 'greedy evaluation'
    
class HeuristicAlternative(Heuristic):

    def __init__(self, initial_state: 'State'):
        super().__init__(initial_state)
        self.v = self.V(initial_state)
        self.k = 1000
    
    def __repr__(self):
        return 'iw alternative'

    def manhattan_distance(self, state: State) -> int:
        return super().manhattan_distance(state)
    def manhattan_distance_with_agent(self, state: State) -> int:
        return super().manhattan_distance_with_agent(state)

    def f(self, state, heuristic_function, state_space):
        # return self.heuristic_qf(state, heuristic_function, state_space)
        # return self.heuristic_qb(state, heuristic_function, state_space)
        # return self.heuristic_qn(state, heuristic_function, state_space)
        return self.heuristic_bn(state, heuristic_function, state_space)

    def heuristic_novelty(self, state_space, fact):
        result = float('inf')
        if fact not in state_space:
            return result
        for h_state in state_space[fact]:
            if h_state < result:
                result = h_state
        return result
    
    def novelty_score_of_fact_in_state(self, fact, state, heuristic_function, state_space):
        return self.heuristic_novelty(state_space, fact) - heuristic_function(state)
    
    def V(self, state):  #afinar esta funcion
        cells = 0
        number_boxes = 0
        number_goals = 0
        v = 0
        for row in range(len(state.boxes)):
            for col in range(len(state.boxes[row])):
                if not State.walls[row][col]:
                    cells += 1
                if state.boxes[row][col] != '':
                    number_boxes += 1
                if State.goals[row][col] != '':
                    number_goals += 1
        v += cells * len(state.agent_rows)
        v += cells * number_boxes
        v += number_goals
        return v
    
    def heuristic_bn(self, state, heuristic_function, state_space):
        for atom in state.atoms:  
            # if str(atom[1]) not in [g for g in self.goal_locations.values()]: # only take important atoms present in goals (so agent that don't have a goal are not counted)
            #     continue
            if self.novelty_score_of_fact_in_state(atom, state, heuristic_function, state_space) > 0:
                return 0
        return 1

    def heuristic_qn(self, state, heuristic_function, state_space):
        heurisicqn = self.v
        nplus = 0
        for atom in state.atoms:
            # if str(atom[1]) not in [g for g in self.goal_locations.values()]: # only take important atoms present in goals (so agent that don't have a goal are not counted)
            #     continue
            if self.novelty_score_of_fact_in_state(atom, state, heuristic_function, state_space) > 0:
                nplus += 1
        heurisicqn -= nplus
        return heurisicqn

    def heuristic_qb(self, state, heuristic_function, state_space):
        heuristicqn = self.heuristic_qn(state, heuristic_function, state_space)
        if heuristicqn < self.v:
            return heuristicqn
        heuristicqb = self.v
        nneg = 0
        for atom in state.atoms:
            # if str(atom[1]) not in [g for g in self.goal_locations.values()]: # only take important atoms present in goals (so agent that don't have a goal are not counted)
            #     continue
            if self.novelty_score_of_fact_in_state(atom, state, heuristic_function, state_space) < 0: 
                nneg += 1
        heuristicqb += nneg
        return heuristicqb
    
    def heuristic_qf(self, state, heuristic_function, state_space):
        nkplus = 0
        for atom in state.atoms:
            if str(atom[1]) not in [g for g in self.goal_locations.values()]: # only take important atoms present in goals (so agent that don's have a goal are not counted)
                continue
            nfs = self.novelty_score_of_fact_in_state(atom, state, heuristic_function, state_space)
            if nfs == float('inf'):
                nkplus += self.k
            elif nfs > 0:
                nkplus = self.k * nfs / max(heuristic_function(state), self.heuristic_novelty(state_space, atom))
        if nkplus > 0:
            return self.k * self.v - nkplus
        else:
            nkneg = 0
            for atom in state.atoms:
                if str(atom[1]) not in [g for g in self.goal_locations.values()]: # only take important atoms present in goals (so agent that don's have a goal are not counted)
                    continue
                nfs = self.novelty_score_of_fact_in_state(atom, state, heuristic_function, state_space)
                if nfs < 0:
                    nkneg += -(self.k * nfs / max(heuristic_function(state), self.heuristic_novelty(state_space, atom)))
            return self.k*self.v + nkneg
        

    def preffered_operators(self, state, heuristic_function):
        num_agents = len(state.agent_rows)
        applicable_actions = [[action for action in Action if state.is_applicable(agent, action)] for agent in range(num_agents)]
        available_actions = []
        joint_action = [None for _ in range(num_agents)]
        actions_permutation = [0 for _ in range(num_agents)]
        
        combinations = list(product(*applicable_actions))
        for comb in combinations:
            if not state.is_conflicting(comb):
                available_actions.append(comb)
        
        preffered_op = SortedList()
        counter = 0
        for o in available_actions:
            h = heuristic_function(state.result(o))
            preffered_op.add((h,counter,o))
            counter +=1
        return preffered_op
    
    def operator_novelty_score(self, operator, state, heuristic_function):
        expanded_states = state.get_expanded_states()
        min_heuristic = float('inf')
        for s in expanded_states:
            if s.joint_action == operator:
                h = heuristic_function(s)
                if h < min_heuristic:
                    min_heuristic = h
        return min_heuristic
    
    def novelty_score_of_operator_in_state(self, operator, state, heuristic_function):
        return self.operator_novelty_score(operator, state, heuristic_function) - heuristic_function(state)
    
    def b_novel_preffered_operators(self,state,heuristic_function, b):
        b_novel = []
        for _,_,o in self.preffered_operators(state, heuristic_function):
            if self.novelty_score_of_operator_in_state(o,state,heuristic_function) > b:
                b_novel.add(o)
        return b_novel
    
    def k_top_b_novel_preffered_operators(self,state,heuristic_function):
        pass

    def max_novel_preffered_operator(self,state,heuristic_function):
        max_preffered_op = None
        max_nov_score = float('-inf')
        for h,_,o in self.preffered_operators(state,heuristic_function):
            n_score = self.novelty_score_of_operator_in_state(o,state,heuristic_function)
            if n_score > max_nov_score:
                max_preffered_op = o
                max_nov_score = n_score
        return max_preffered_op    
    


class Heuristicderelax(Heuristic):

    def __init__(self, initial_state: 'State'):
        super().__init__(initial_state)
        self.goal_locations = {}
        self.only_goals = set()
        self.all_distances = {}
        # self.compute_goals(initial_state)

    def return_goals(self):
        return self.only_goals

    def f(self, state: 'State') -> 'int':
        # self.compute_goals(state)
        result = []
        result.append(self.manhattan_distance_avoiding_walls(state)**2)
        result.append(self.penalization_based_on_goals(state))
        # result.append(self.goals_not_achieved(state))
        # print('the results are', result, flush=True)
        # result += state.g
        return sum(result)
    
    def compute_goals(self, state):
        self.goal_locations = {}
        self.only_goals = set()
        self.all_distances = {}
        for row in range(len(State.goals)):
            for col in range(len(State.goals[row])):
                if State.goals[row][col] != '':
                    goal = State.goals[row][col]
                    self.goal_locations[(row,col)] = goal
                    self.only_goals.add(goal)
                self.all_distances[(row,col)] = self.precompute_distance_map(state, None, row,col)
        self.ordered_goals = self.ordered_goals_based_on_difficulty(state)
        self.ordered_goals = sorted(self.ordered_goals,  key=lambda x:(x[0]+x[1], x[0],x[1]) , reverse=True)
        self.ordered_goals = sorted(self.ordered_goals,  key=lambda x:(x[0]+x[1], x[0],x[1],-(abs(x[3] - self.ordered_goals[0][3]) + abs(x[4] - self.ordered_goals[0][4]))) , reverse=True)
        # print('#Finish preprocessing', flush=True)
        # print(self.ordered_goals, flush=True)

    def penalization_based_on_goals(self, state: 'State'):
        result = 0
        only_goals = [(g[2],g[3],g[4]) for g in self.ordered_goals]
        agent_goal_only_intheend = True

        for (row,col),goal in self.goal_locations.items():
            if '0' <= goal <= '9':
                agent, arow, acol, dist = self.find_agent_based_on_goal(state, goal, row, col)
                penalty = 1
                pos_in_priority = (len(self.ordered_goals) - only_goals.index((goal,row,col)))+1
                if agent_goal_only_intheend and self.agent_boxes_all_in_goals(state, agent, row, col):                
                    for (walls2, near_goals, goal2, row2, col2) in self.ordered_goals:
                        if (row,col) == (row2,col2) and walls2 >= self.max_walls_not_achieved_in_goal(state):
                            if (state.agent_rows[int(agent)],state.agent_cols[int(agent)]) != (row,col) and self.goal_blocked(state, goal,row,col):
                                penalty = math.log(walls2)
                            break
                    result += dist * (pos_in_priority**2) * penalty

            elif 'A' <= goal <= 'Z':
                penalty = 1
                closest_box = self.find_box_based_on_goal(state, goal,row,col)
                pos_in_priority = (len(self.ordered_goals) - only_goals.index((goal,row,col)))+1
                dist = closest_box[3] 
                for (walls2, near_goals, goal2, row2, col2) in self.ordered_goals:
                    if (row,col) == (row2,col2):
                        if state.joint_action != None and any(action.type in [ActionType.Push, ActionType.Pull] for action in state.joint_action):
                            break
                        if walls2 == self.max_walls_not_achieved_in_goal(state) and state.boxes[row][col] == '' and self.goal_blocked(state,goal,row,col):
                            if walls2+near_goals == 0:
                                continue
                            penalty = math.log(walls2+near_goals)
                            break
                result += dist * (pos_in_priority**2) * penalty
        return result

    def manhattan_distance_avoiding_walls(self, state: 'State') -> 'int':
        
        result = 0
        distance_agent_box_reduced_by = 1000
        agent_goal_only_in_the_end = True

        for (row,col),goal in self.goal_locations.items():
            if '0' <= goal <= '9':
                agent = ord(goal) - ord('0')
                distance_map = self.all_distances[(row,col)]
                agent_row, agent_col = state.agent_rows[agent], state.agent_cols[agent]
                dist = distance_map[agent_row][agent_col]
                if agent_goal_only_in_the_end and self.agent_boxes_all_in_goals(state, agent, row, col):
                    result += dist
            elif 'A' <= goal <= 'Z':
                distance_map = self.all_distances[(row,col)]
                closest_box = self.find_box_based_on_goal(state, goal, row, col)
                box, box_row, box_col = closest_box[0], closest_box[1], closest_box[2]
                dist = distance_map[box_row][box_col]
                result += dist

        if distance_agent_box_reduced_by > 0:
            for agent in range(len(state.agent_rows)):
                agent, agent_row, agent_col = str(agent), state.agent_rows[agent], state.agent_cols[agent]
                box_info = self.find_box_based_on_agent(state, agent, agent_row, agent_col)
                if box_info is not None:
                    box, box_row, box_col,distance = box_info
                    result += distance/distance_agent_box_reduced_by
        return result
    

    def distance_to_path(self, state, paths):
        # print(paths, flush=True)
        result = 0
        for agent in range(len(state.agent_rows)):
            position = (state.agent_rows[agent], state.agent_cols[agent])
            min_dist = float('inf')
            for i,p in enumerate(paths[agent]):
                dist = self.all_distances[position][p[0]][p[1]]
                # dist *= len(paths[agent]) - i**2 + 1
                if dist < min_dist:
                    min_dist = dist
            result += min_dist
        return result

    def distance_to_path_boxes(self, state, paths, priority):
        # print(paths, flush=True)
        result = 0
        box_pos = {b:(r,c) for (b,_,r,c) in self.get_boxes(state)}
        p_0 = [p[0] for p in priority]
        for box in state.paths.keys():
            pos = box_pos[box]
            if State.goals[box_pos[box][0]][box_pos[box][1]] == box:
                continue

            distances = []
            for i,p in enumerate(state.paths[box]):
                dist = self.all_distances[pos][p[0]][p[1]]
                distances.append((dist,len(state.paths[box])-i))

            penalty = 1
            for (b,r,c) in priority:
                if p_0.index(box) > p_0.index(b) and State.goals[box_pos[b][0]][box_pos[b][1]] != b:
                    penalty = 2#(len(priority)-priority.index(b))**2

            distances = sorted(distances, key=lambda x: (x[0],x[1]))
            min_dist = distances[0] 
            prior = len(priority) - p_0.index(box)
            result += (min_dist[0]+1) * (prior**2) * penalty

        return result

    def distance_box_to_selected_goals(self, state):
        selections = state.selections
        result = 0
        for box_info, goal_info in selections.items():
            (bow, brow, bcol), (goal, grow, gcol) = box_info, goal_info
            dist = self.all_distances[(brow,bcol)][grow][gcol]
            result += math.log(dist+1)
        return result


    def boxes_not_in_their_respective_goals(self, state):
        selections = state.selections
        result = len(selections)
        print('aaa', selections, flush=True)
        for box_info, goal_info in selections.items():
            (bow, brow, bcol), (goal, grow, gcol) = box_info, goal_info
            if (grow, gcol) == (brow, bcol) : 
                result -= 1
        return result


    def goal_closest_to_goal(self, state):
        assignments = {}
        for row in range(len(state.boxes)):
            for col in range(len(state.boxes[row])):
                box = state.boxes[row][col]
                if box != '':
                    goals = []
                    for grow in range(len(State.goals)):
                        for gcol in range(len(State.goals[grow])):
                            if State.goals[grow][gcol] == box:
                                goal = State.goals[grow][gcol]
                                goals.append((goal, grow, gcol, self.all_distances[(row,col)][grow][gcol]))
                    closest_goal = sorted(goals, key=lambda x: x[3])[0]
                    assignments[(box,row,col)] = (closest_goal[0], closest_goal[1], closest_goal[2])
        return assignments 


    def find_goals_near_goal(self, state, goal, row, col):
        goals = {}
        for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            if State.goals[row+dx][col+dy] != '':
                goals[State.goals[row+dx][col+dy]] = [row+dx,col+dy]
        return goals

    def goal_blocked(self, state, goal, grow, gcol):
        near_goals = self.find_goals_near_goal(state, goal, grow,gcol)
        ordered_goals = [(g[2],g[3],g[4]) for g in self.ordered_goals]
        goal_index = ordered_goals.index((goal,grow,gcol))
        walls_near = self.ordered_goals[goal_index][0]
        # if walls_near + len(near_goals) < 8:
        #     return False
        for g,[r,c] in near_goals.items():
            gindex = ordered_goals.index((g,r,c))
            if state.boxes[r][c] != g and goal_index < gindex:
                return False
        return True

    def is_respect_priority(self, state, goal, grow,gcol):
        ordered_goals = [(g[2],g[3],g[4]) for g in self.ordered_goals]
        goal_index = ordered_goals.index((goal,grow,gcol))
        for i,(_,_,g,r,c) in enumerate(self.ordered_goals):
            if i > goal_index:
                if '0' <= g <= '9' and state.agent_rows[ord(str(g))-ord('0')]==r and state.agent_cols[ord(str(g))-ord('0')]==c: 
                    return False
                if 'A' <= g <= 'Z' and state.boxes[r][c] == g:
                    return False
        return True

    def precompute_distance_map(self, state, goal, goal_row, goal_col):
        max_rows = len(state.boxes)
        max_cols = len(state.boxes[0])
        distance_map = [[float('inf')]*max_cols for _ in range(max_rows)]
        queue = deque([(goal_row, goal_col, 0)])
        while queue:
            row,col,dist = queue.popleft()
            if distance_map[row][col] != float('inf'):
                continue
            distance_map[row][col] = dist
            for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                neighbor = (row + dx, col + dy)
                if 0 <= neighbor[0] < max_rows and 0 <= neighbor[1] < max_cols and not State.walls[neighbor[0]][neighbor[1]]:
                    queue.append((neighbor[0],neighbor[1], dist+1))
        return distance_map

    def precompute_distance_map_with_goals_as_walls(self, state, goal, goal_row, goal_col):
        max_rows = len(state.boxes)
        max_cols = len(state.boxes[0])
        distance_map = [[float('inf')]*max_cols for _ in range(max_rows)]
        queue = deque([(goal_row, goal_col, 0)])
        while queue:
            row,col,dist = queue.popleft()
            if distance_map[row][col] != float('inf'):
                continue
            distance_map[row][col] = dist
            for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                neighbor = (row + dx, col + dy)
                if 0 <= neighbor[0] < max_rows and 0 <= neighbor[1] < max_cols:
                    if not State.walls[neighbor[0]][neighbor[1]] and (State.goals[neighbor[0]][neighbor[1]] == '' or state.boxes[neighbor[0]][neighbor[1]] == '' or state.boxes[neighbor[0]][neighbor[1]] != State.goals[neighbor[0]][neighbor[1]]):
                        queue.append((neighbor[0],neighbor[1], dist+1))
        return distance_map

    def ordered_goals_based_on_difficulty(self,state: 'State'):
        queue = []
        for (row,col),goal in self.goal_locations.items():
            walls_near = 0
            goals_near = 0
            free_spaces = 0
            # for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            for dx in [-1,0,1]:
                for dy in [-1,0,1]:
                    if dx == 0 and dy == 0:
                        continue
                    if State.walls[row+dx][col+dy]:
                        walls_near += 1
                    if State.goals[row+dx][col+dy] != '':
                        goals_near += 1
            heapq.heappush(queue, (walls_near, goals_near, goal, row, col))
        return queue
    
    def goals_not_achieved(self, state: 'State'):
        result = len(self.goal_locations)
        for (row,col),goal in self.goal_locations.items():
            if '0' <= goal <= '9' and state.agent_rows[ord(goal)-ord('0')] == row and state.agent_cols[ord(goal)-ord('0')] == col:
                result -= 1
            elif 'A' <= goal <= 'Z' and state.boxes[row][col] == goal:
                result -= 1
        return result
    
    def goals_achieved(self, state: 'State'):
        result = 0
        for (row,col),goal in self.goal_locations.items():
            if '0' <= goal <= '9' and state.agent_rows[ord(goal)-ord('0')] == row and state.agent_cols[ord(goal)-ord('0')] == col:
                result += 1
            elif 'A' <= goal <= 'Z' and state.boxes[row][col] == goal:
                result += 1
        return result
    
    def goals_achieved_per_agent(self, state):
        results = {}
        for agent in range(len(state.agent_rows)):
            result = 0
            agent_color = State.agent_colors[agent]
            for (row,col),goal in self.goal_locations.items():
                if '0' <= goal <= '9' and goal == str(agent) and state.agent_rows[ord(goal)-ord('0')] == row and state.agent_cols[ord(goal)-ord('0')] == col:
                    result += 1
                elif 'A' <= goal <= 'Z' and State.box_colors[ord(goal)-ord('A')]==agent_color and state.boxes[row][col] == goal:
                    result += 1  
            results[agent] = result
        return results

    def which_goals_are_achieved(self, state: 'State'):
        goals_achieved = []
        for row in range(len(State.goals)):
            for col in range(len(State.goals[row])):
                if State.goals[row][col] != '':
                    goal = State.goals[row][col]
                    if 'A' <= goal <= 'Z' and state.boxes[row][col] == goal:
                        goals_achieved.append((goal,row,col))
                    elif '0' <= goal <= '9' and state.agent_rows[ord(goal)-ord('0')] == row and state.agent_cols[ord(goal)-ord('0')] == col:
                        goals_achieved.append((goal,row,col))
        return goals_achieved

    def find_box_based_on_goal(self, state : 'State', goal, grow, gcol):
        boxes = []
        distance_map = self.all_distances[(grow,gcol)]
        for row in range(len(state.boxes)):
            for col in range(len(state.boxes[row])):
                if state.boxes[row][col] == goal:# and State.goals[row][col] == '':
                    boxes.append((state.boxes[row][col],row,col, distance_map[row][col])) 
        return sorted(boxes, key=lambda x: x[3])[0]
    
    def find_agent_based_on_box(self, state: 'State', box, row, col):
        agents = []
        for ag in range(len(state.agent_rows)):
            agent_color = State.agent_colors[ag]
            box_color = State.box_colors[ord(box) - ord('A')]
            if agent_color == box_color:
                agent_row, agent_col = state.agent_rows[ag], state.agent_cols[ag]
                dist = math.sqrt((agent_row - row) ** 2 + (agent_col - col) ** 2)
                agents.append((ag, agent_row, agent_col,dist))
        return sorted(agents, key=lambda x:x[3])[0]
    
    def find_box_based_on_agent(self, state,agent, arow, acol):
        boxes = []
        agent_color = State.agent_colors[ord(str(agent))-ord('0')]
        for row in range(len(state.boxes)):
            for col in range(len(state.boxes[row])):
                box = state.boxes[row][col]
                if box != '' and State.box_colors[ord(box)-ord('A')] == agent_color and State.goals[row][col] != box:
                    dist = self.all_distances[(arow,acol)][row][col]
                    boxes.append([box, row, col, dist])
        if boxes == []:
            return None
        return sorted(boxes, key=lambda x:x[3])[0]
    
    def find_agent_based_on_goal(self, state, goal, grow, gcol):
        agent = ord(str(goal)) - ord('0')
        agent_row, agent_col = state.agent_rows[agent], state.agent_cols[agent],
        return (agent, agent_row, agent_col, self.all_distances[(grow,gcol)][agent_row][agent_col])

    def agent_boxes_all_in_goals(self, state, agent, arow, acol):
        agent_color = State.agent_colors[ord(str(agent))-ord('0')]
        for row in range(len(state.boxes)):
            for col in range(len(state.boxes[row])):
                if state.boxes[row][col]!='':
                    box = state.boxes[row][col]
                    if State.box_colors[ord(box)-ord('A')] == agent_color:
                        if State.goals[row][col] == '' or self.goal_locations[(row,col)] != box:
                            return False
        return True

    def max_walls_not_achieved_in_goal(self, state):
        all_walls = []
        for (walls, _, goal, row, col) in self.ordered_goals:
            if state.boxes[row][col] != goal:
                all_walls.append(walls)
        if all_walls==[]:
            return 1
        return max(all_walls)

    def distance_agent_goal(self,state,agent,arow,acol):
        for (row,col),goal in self.goal_locations.items():
            if goal == agent:
                return self.all_distances[(arow,acol)][row][col]

    def get_g(self, state):
        return state.g
    
    def get_new_atoms(self,state):
        return state.new_atoms
    
    def get_boxes(self, state):
        boxes = []
        for row in range(len(state.boxes)):
            for col in range(len(state.boxes[row])):
                if state.boxes[row][col] != '':
                    box = state.boxes[row][col]
                    boxes.append((box, State.box_colors[ord(box)-ord('A')], row, col))
        return boxes
    
    def get_goals(self, state):
        goals = []
        for row in range(len(State.goals)):
            for col in range(len(State.goals[row])):
                if State.goals[row][col] != '':
                    goal= State.goals[row][col]
                    goals.append((goal,row,col))
        return goals
    




    def agent_is_moving_a_box(self,state):
        if state.joint_action == None:
            return True
        if any(action.type in [ActionType.Push, ActionType.Pull] for action in state.joint_action):
            return True
        return False

    def only_manhattan_distance_avoiding_walls(self, state: 'State') -> 'int':
        result = 0
        agent_goal_only_intheend = True

        for (row,col),goal in self.goal_locations.items():
            if '0' <= goal <= '9':
                agent = ord(goal) - ord('0')
                distance_map = self.all_distances[(row,col)]
                agent_row, agent_col = state.agent_rows[agent], state.agent_cols[agent]
                dist = distance_map[agent_row][agent_col]
                if agent_goal_only_intheend and self.agent_boxes_all_in_goals(state, agent, row, col):
                    result += dist
            elif 'A' <= goal <= 'Z':
                distance_map = self.all_distances[(row,col)]
                closest_box = self.find_box_based_on_goal(state, goal, row, col)
                box, box_row, box_col = closest_box[0], closest_box[1], closest_box[2]
                dist = distance_map[box_row][box_col]
                result += dist
        return result

    def get_random(self,state):
        return random.randint(1,10000)

    def __repr__(self):
        return 'Relaxation'


class HeuristicOba(Heuristic):
    def __init__(self, initial_state: 'State'):
        super().__init__(initial_state)
        # self.goal_locations = {}
        # self.only_goals = set()
        # self.all_distances = {}
        # self.all_boxes = {}
        # for row in range(len(State.goals)):
        #     for col in range(len(State.goals[row])):
        #         if State.goals[row][col] != '':
        #             goal = State.goals[row][col]
        #             self.goal_locations[(row,col)] = goal
        #             self.only_goals.add(goal)
        #         if initial_state.boxes[row][col] != '':
        #             self.all_boxes[(row,col)] = initial_state.boxes[row][col]
        #         self.all_distances[(row,col)] = self.precompute_distance_map(initial_state, None, row,col)

    def __repr__(self):
        return 'OBA'


    def precompute_distance_map(self, state, goal, goal_row, goal_col):
        max_rows = len(state.boxes)
        max_cols = len(state.boxes[0])
        distance_map = [[float('inf')]*max_cols for _ in range(max_rows)]
        queue = deque([(goal_row, goal_col, 0)])
        while queue:
            row,col,dist = queue.popleft()
            if distance_map[row][col] != float('inf'):
                continue
            distance_map[row][col] = dist
            for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                neighbor = (row + dx, col + dy)
                if 0 <= neighbor[0] < max_rows and 0 <= neighbor[1] < max_cols and not State.walls[neighbor[0]][neighbor[1]]:
                    queue.append((neighbor[0],neighbor[1], dist+1))
        return distance_map


    def get_goals(self):
        return self.only_goals
    
    def get_initial_mg(self):
        m_g = set()
        goals = self.goal_locations
        boxes = self.all_boxes
        if len(boxes) == 0 or all([ord('0') <= ord(g) <= ord('9') for g in goals.values()]):
            return m_g
        for goal in list(set(goals.values())):
            gs = [(rg,cg,g) for (rg,cg),g in goals.items() if g == goal]
            bs = [(rb,cb,b) for (rb,cb),b in boxes.items() if b == goal]
            if len(gs) == len(bs):
                for (rb,cb,b) in bs:
                    m_g.add((rb,cb,b))
                continue
            # There are more boxes than goals
            all_goals = gs
            all_boxes = bs
            for (rg,cg,g) in all_goals:
                closest_box = None
                dist = float('inf')
                for (r,c,b) in all_boxes:
                    if self.all_distances[(r,c)][rg][cg] < dist:
                        closest_box = (r,c,b)
                        dist = self.all_distances[(r,c)][rg][cg]
                m_g.add(closest_box)
                all_boxes.remove(closest_box)
        return m_g


    def f(self, state: 'State') -> 'int':
        # self.compute_goals(state)
        return 1

