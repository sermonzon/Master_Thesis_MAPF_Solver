import random
import copy
from action import Action, ActionType
from itertools import product
from collections import deque
import heapq
import string
import bisect
import itertools
import math



class State:
    _RNG = random.Random(1)
    all_distances = None
    goal_locations = None
    found_all_distances_and_goal_locations = False
    
    def __init__(self, agent_rows, agent_cols, boxes):
        '''
        Constructs an initial state.
        Arguments are not copied, and therefore should not be modified after being passed in.
        
        The lists walls, boxes, and goals are indexed from top-left of the level, row-major order (row, col).
               Col 0  Col 1  Col 2  Col 3
        Row 0: (0,0)  (0,1)  (0,2)  (0,3)  ...
        Row 1: (1,0)  (1,1)  (1,2)  (1,3)  ...
        Row 2: (2,0)  (2,1)  (2,2)  (2,3)  ...
        ...
        
        For example, State.walls[2] is a list of booleans for the third row.
        State.walls[row][col] is True if there's a wall at (row, col).
        
        The agent rows and columns are indexed by the agent number.
        For example, State.agent_rows[0] is the row location of agent '0'.
                
        Note: The state should be considered immutable after it has been hashed, e.g. added to a dictionary or set.
        '''
        self.agent_rows = agent_rows
        self.agent_cols = agent_cols
        self.boxes = boxes
        self.parent = None
        self.joint_action = None
        self.g = 0
        self._hash = None
        self.atoms = self.update_atoms()
        self.fixed_agents = set()
        self.new_atoms = 0
        self.iw = 1
        self.paths = {}
        self.selections = {}
        self.oba_mg = set()
        self.qqueue = []
        self.rqueue = []
        self.lenq = 2
        self.lenr = 2
        
        if not State.found_all_distances_and_goal_locations:
            print('starttt',flush=True)
            State.found_all_distances_and_goal_locations = True
            goal_locations = {}
            State.all_distances = {}
            self.boxes_locations = set()

            for row in range(len(State.goals)):
                for col in range(len(State.goals[row])):
                    if State.goals[row][col] != '':
                        goal = State.goals[row][col]
                        goal_locations[(row,col)] = goal
                    State.all_distances[(row,col)] = self.precompute_distance_map(row,col)
                    if self.boxes[row][col] != '':
                        self.boxes_locations.add((row,col,self.boxes[row][col]))
            State.goal_locations = goal_locations
            print('finishhh',flush=True)


        # if State.goal_locations == None:
            # State.goal_locations = self.get_goal_locations()
        # if State.all_distances == None:
        #     State.all_distances = {}
        #     for row in range(len(State.goals)):
        #         for col in range(len(State.goals[row])):
        #             State.all_distances[(row,col)] = self.precompute_distance_map(row,col)
        # self.boxes_locations = set()
        # for row in range(len(self.boxes)):
        #     for col in range(len(self.boxes[row])):
        #         if self.boxes[row][col] != '':
        #             self.boxes_locations.add((row,col,self.boxes[row][col]))


    # For Renew Atoms, visit this function 
    def result(self, joint_action: '[Action, ...]') -> 'State':
        '''
        Returns the state resulting from applying joint_action in this state.
        Precondition: Joint action must be applicable and non-conflicting in this state.
        '''
        
        # Copy this state.
        copy_agent_rows = self.agent_rows[:]
        copy_agent_cols = self.agent_cols[:]
        copy_boxes = [row[:] for row in self.boxes]
        new_selections = self.selections.copy()
        boxes_locations = set(self.boxes_locations)
        
        # Apply each action.
        for agent, action in enumerate(joint_action):
            if action.type is ActionType.NoOp:
                pass
            
            elif action.type is ActionType.Move:
                copy_agent_rows[agent] += action.agent_row_delta
                copy_agent_cols[agent] += action.agent_col_delta

            elif action.type is ActionType.Push:
                copy_agent_rows[agent] += action.agent_row_delta
                copy_agent_cols[agent] += action.agent_col_delta
                box_row = copy_agent_rows[agent]
                box_col = copy_agent_cols[agent]
                box = copy_boxes[box_row][box_col]
                copy_boxes[box_row][box_col] = ''
                copy_boxes[box_row + action.box_row_delta][box_col + action.box_col_delta] = box

                boxes_locations.remove((box_row, box_col, box))
                boxes_locations.add((box_row + action.box_row_delta,box_col + action.box_col_delta, box))


                if new_selections != {}:
                    (boxx, brow, bcol) = (box, box_row, box_col)
                    (new_box, new_brow, new_bcol) = (boxx, box_row + action.box_row_delta,box_col + action.box_col_delta)
                    if (boxx, brow, bcol) in new_selections:
                        (goal, grow, gcol) = new_selections[(boxx, brow, bcol)]
                        del new_selections[(boxx, brow, bcol)]
                        new_selections[(new_box, new_brow, new_bcol)] = (goal, grow, gcol)

            
            elif action.type is ActionType.Pull:
                box_row = copy_agent_rows[agent] - action.box_row_delta
                box_col = copy_agent_cols[agent] - action.box_col_delta
                box = copy_boxes[box_row][box_col]
                copy_boxes[box_row][box_col] = ''
                copy_boxes[copy_agent_rows[agent]][copy_agent_cols[agent]] = box

                boxes_locations.remove((box_row, box_col, box))
                boxes_locations.add((copy_agent_rows[agent],copy_agent_cols[agent], box))

                if new_selections != {}:
                    (boxx, brow, bcol) = (box, box_row, box_col)
                    (new_box, new_brow, new_bcol) = (boxx, copy_agent_rows[agent],copy_agent_cols[agent])
                    if (boxx, brow, bcol) in new_selections:    
                        (goal, grow, gcol) = new_selections[(boxx, brow, bcol)]
                        del new_selections[(boxx, brow, bcol)]
                        new_selections[(new_box, new_brow, new_bcol)] = (goal, grow, gcol)

                copy_agent_rows[agent] += action.agent_row_delta
                copy_agent_cols[agent] += action.agent_col_delta

            
        copy_state = State(copy_agent_rows, copy_agent_cols, copy_boxes)
        
        copy_state.parent = self
        copy_state.joint_action = joint_action[:]
        copy_state.g = self.g + 1


        # For RenewAtoms, change use_renew_atoms to True, else select it False
        use_renew_atoms = False
        if not use_renew_atoms:        
            copy_state.atoms = copy_state.update_atoms()
        else:
            m = 50 # Change parameter m as desired
            copy_state.atoms = copy_state.update_atoms_renew_atoms(m)

        copy_state.fixed_agents = self.fixed_agents
        copy_state.paths = self.paths
        copy_state.selections = new_selections
        copy_state.oba_mg = self.oba_mg
        copy_state.boxes_locations = boxes_locations
        copy_state.new_atoms = self.new_atoms
        
        return copy_state
    
    def is_goal_state(self) -> 'bool':
        for row in range(len(State.goals)):
            for col in range(len(State.goals[row])):
                goal = State.goals[row][col]
                
                if 'A' <= goal <= 'Z' and self.boxes[row][col] != goal:
                    return False
                elif '0' <= goal <= '9' and not (self.agent_rows[ord(goal) - ord('0')] == row and self.agent_cols[ord(goal) - ord('0')] == col):
                    return False
        return True
    
    def goals_achieved(self) -> 'int':
        result=0
        for row in range(len(State.goals)):
            for col in range(len(State.goals[row])):
                goal = State.goals[row][col]
                if 'A' <= goal <= 'Z' and self.boxes[row][col] == goal:
                    result+=1
                elif '0' <= goal <= '9' and self.agent_rows[ord(goal)-ord('0')] == row and self.agent_cols[ord(goal)-ord('0')] == col:
                    result+=1
        return result
    

    # To control how to expand the states, change the parameters inside this function. 
    def get_expanded_states(self) -> '[State, ...]':
        num_agents = len(self.agent_rows)
        
        applicable_actions = [[action for action in Action if self.is_applicable(agent, action)] for agent in range(num_agents)]

        # Select controlling_expansion = False and method_random = True to not apply any controlling expansion strategy
        # Select controlling_expansion = True and method_random = True to apply controlling expansion using Random Method
        # Select controlling_expansion = True and method_random = False to apply controlling expansion using Heuristic Method
        controlling_expansion = True
        method_random = False
        
        if not method_random:
            expanded_states = []
            counter = 0
            for joint_action in product(*applicable_actions):
                if not self.is_conflicting(joint_action):
                    state = self.result(joint_action)
                    bisect.insort(expanded_states, (self.manhattan_distance_avoiding_walls(state, method_focus_on_actions=False), counter, state))
                counter += 1
            if controlling_expansion:
                length = sum([len(a) for a in applicable_actions])
                return [item[2] for item in expanded_states[:length]]
            else:
                return [item[2] for item in expanded_states]
            
        elif method_random:
            expanded_states = [
                self.result(joint_action)
                for joint_action in product(*applicable_actions)
                    if not self.is_conflicting(joint_action)
            ]
            State._RNG.shuffle(expanded_states)

            if controlling_expansion:
                length = sum([len(a) for a in applicable_actions])
                return expanded_states[:length] if length < len(expanded_states) else expanded_states
            else:
                return expanded_states
            


    def get_expanded_states_version_2(self) -> '[State, ...]':
        # print('#empezamos',flush=True)
        num_agents = len(self.agent_rows)
        
        applicable_actions = [[action for action in Action if self.is_applicable(agent, action)] for agent in range(num_agents)]
        # print('#eeee', applicable_actions, flush=True)
        
        counter = 0
        all_actions = [[] for agent in range(num_agents)]
        for agent in range(num_agents):
            noops = [Action.NoOp for agent in range(num_agents)]
            best_actions = []
            # State._RNG.shuffle(applicable_actions[agent])
            for action in applicable_actions[agent]:
                noops[agent] = action
                if not self.is_conflicting(noops):
                    state = self.result(noops)
                    bisect.insort(best_actions, (self.goals_not_achieved_with_state(state),self.manhattan_distance_avoiding_walls(state, method_focus_on_actions=False), counter, action))
                    counter += 1
            # print('#rrrr', agent, best_actions, flush=True)
            for _,_,_,act in best_actions:
                all_actions[agent].append(act)
        # print('#aaaa', all_actions,flush=True)
            

        length = sum([len(a) for a in applicable_actions])
        indices = [range(0,len(act)) for act in all_actions]
        prod_generator = itertools.product(*indices)
        prod = heapq.nsmallest(length, prod_generator, key=sum)
        
        # print('pppp',prod,flush=True)
        

        # print('#bbb', prod,flush=True)
        joint_actions = []
        for index in prod:
            actions = []
            for i,ind in enumerate(index):
                actions.append(all_actions[i][ind])
            joint_actions.append(actions)
        # print([[act.name for act in j] for j in joint_actions], flush=True)
        # print('#cccc', joint_actions,flush=True)
        expanded_states = [
                self.result(joint_action)
                for joint_action in joint_actions
                    if not self.is_conflicting(joint_action)
        ]
        # print('aaaa', length, len(joint_actions),len(expanded_states), flush=True)
        
        return expanded_states




    def is_applicable(self, agent: 'int', action: 'Action') -> 'bool':
        agent_row = self.agent_rows[agent]
        agent_col = self.agent_cols[agent]
        agent_color = State.agent_colors[agent]
        
        # if agent in self.fixed_agents:
        #     if action.type is ActionType.NoOp:
        #         return True
        #     else: 
        #         return False

        if action.type is ActionType.NoOp:
            return True
            
        elif action.type is ActionType.Move:
            destination_row = agent_row + action.agent_row_delta
            destination_col = agent_col + action.agent_col_delta
            return self.is_free(destination_row, destination_col)
        
        elif action.type is ActionType.Push:
            destination_row_agent = agent_row + action.agent_row_delta
            destination_col_agent = agent_col + action.agent_col_delta
            box_row = destination_row_agent
            box_col = destination_col_agent
            destination_row_box = box_row + action.box_row_delta
            destination_col_box = box_col + action.box_col_delta
            box = self.box_at(box_row,box_col)
            return (box != '') and (self.match_colors(agent, box)) and self.is_free(destination_row_box,destination_col_box) 
                
        elif action.type is ActionType.Pull:
            destination_row_agent = agent_row + action.agent_row_delta
            destination_col_agent = agent_col + action.agent_col_delta
            box_row = agent_row - action.box_row_delta
            box_col = agent_col - action.box_col_delta
            destination_row_box = agent_row
            destination_col_box = agent_col
            box = self.box_at(box_row,box_col)
            return (box != '') and (self.match_colors(agent,box)) and self.is_free(destination_row_agent,destination_col_agent)

    def is_conflicting(self, joint_action: '[Action, ...]') -> 'bool': # type: ignore
        num_agents = len(self.agent_rows)

        # agent_rows = [None for _ in range(num_agents)]
        # agent_cols = [None for _ in range(num_agents)]
        destination_rows = [None for _ in range(num_agents)] # row of new cell to become occupied by action
        destination_cols = [None for _ in range(num_agents)] # column of new cell to become occupied by action
        box_rows = [None for _ in range(num_agents)] # current row of box moved by action
        box_cols = [None for _ in range(num_agents)] # current column of box moved by action
        destination_box_rows = [None for _ in range(num_agents)]
        destination_box_cols = [None for _ in range(num_agents)]
        
        # Collect cells to be occupied and boxes to be moved.
        for agent in range(num_agents):
            action = joint_action[agent]
            agent_row = self.agent_rows[agent]
            agent_col = self.agent_cols[agent]
            
            if action.type is ActionType.NoOp:
                pass
             
            if action.type == ActionType.Move:
                destination_rows[agent] = agent_row + action.agent_row_delta
                destination_cols[agent] = agent_col + action.agent_col_delta

            elif action.type == ActionType.Push:
                box_row = agent_row + action.agent_row_delta
                box_col = agent_col + action.agent_col_delta
                box_rows[agent] = box_row
                box_cols[agent] = box_col
                new_box_row = box_row + action.box_row_delta
                new_box_col = box_col + action.box_col_delta
                destination_rows[agent] = box_row
                destination_cols[agent] = box_col
                destination_box_rows[agent] = new_box_row
                destination_box_cols[agent] = new_box_col

            elif action.type == ActionType.Pull:
                destination_rows[agent] = agent_row + action.agent_row_delta
                destination_cols[agent] = agent_col + action.agent_col_delta
                box_row = agent_row - action.box_row_delta  # Box moves to the agent's original position
                box_col = agent_col - action.box_col_delta
                box_rows[agent] = box_row
                box_cols[agent] = box_col
                destination_box_rows[agent] = agent_row
                destination_box_cols[agent] = agent_col


        for a1 in range(num_agents):
            if joint_action[a1].type == ActionType.NoOp:
                continue
            
            for a2 in range(a1 + 1, num_agents):
                if joint_action[a2].type == ActionType.NoOp:
                    continue
                
                # Moving into same cell?
                if destination_rows[a1] == destination_rows[a2] and destination_cols[a1] == destination_cols[a2]:
                    return True
                #Exchanging cells
                if (destination_rows[a1] == self.agent_rows[a2] and destination_cols[a1] == self.agent_cols[a2]) and (destination_rows[a2] == self.agent_rows[a1] and destination_cols[a2] == self.agent_cols[a1]):
                    return True
                # Agent moving to destination of box
                if destination_rows[a1] == destination_box_rows[a2] and destination_cols[a1] == destination_box_cols[a2]:
                    return True
                if destination_rows[a2] == destination_box_rows[a1] and destination_cols[a2] == destination_box_cols[a1]:
                    return True
                # 2 agent moving the same box
                if box_rows[a1] != None and box_cols[a1] != None and box_rows[a2] != None and box_cols[a2] != None:
                    if box_rows[a1] == box_rows[a2] and box_cols[a1] == box_cols[a2]:
                        return True
                    
                # 2 boxes into same cell
                if destination_box_rows[a1] is not None and destination_box_rows[a2] is not None:
                    if destination_box_rows[a1] == destination_box_rows[a2] and destination_box_cols[a1] == destination_box_cols[a2]:
                        return True

        return False






    def is_free(self, row: 'int', col: 'int') -> 'bool':
        return not State.walls[row][col] and self.boxes[row][col] == '' and self.agent_at(row, col) is None
    
    def agent_at(self, row: 'int', col: 'int') -> 'char':
        for agent in range(len(self.agent_rows)):
            if self.agent_rows[agent] == row and self.agent_cols[agent] == col:
                return chr(agent + ord('0'))
        return None
    
    def box_at(self, row: 'int', col: 'int'):
        return self.boxes[row][col]
    
    def match_colors(self, agent, box):
        return State.box_colors[ord(box) - ord('A')] == State.agent_colors[ord(str(agent)) - ord('0')]

    def extract_plan(self) -> '[Action, ...]':
        plan = [None for _ in range(self.g)]
        state = self
        while state.joint_action is not None:
            plan[state.g - 1] = state.joint_action
            state = state.parent
        return plan
    
    def update_atoms(self):
        self.atoms = set()
        
        for agent in range(len(self.agent_rows)):
            atom = ('agent',agent, State.agent_colors[agent], self.agent_rows[agent], self.agent_cols[agent])
            self.atoms.add(atom)
            if str(agent) == State.goals[self.agent_rows[agent]][self.agent_cols[agent]]:
                atom = ('goal', agent, State.agent_colors[agent], self.agent_rows[agent], self.agent_cols[agent])
                self.atoms.add(atom)
        
        for row in range(len(self.boxes)):
            for col in range(len(self.boxes[row])):
                if self.boxes[row][col] != '':
                    box = self.boxes[row][col]
                    atom = ('box', box, State.box_colors[ord(box) - ord('A')], row,col)
                    self.atoms.add(atom)
                    if State.goals[row][col] == box:
                        atom = ('goal',box,row,col)
                        self.atoms.add(atom)
        
        return self.atoms

    # This function creates the atoms based on agent-action pairs
    def update_atoms_renew_atoms(self,m):
        self.atoms = set()
        for agent,action in enumerate(self.joint_action):
                # prev_row, prev_col = self.agent_rows[agent] - action.agent_row_delta, self.agent_cols[agent] - action.agent_col_delta
                act_row, act_col = self.agent_rows[agent], self.agent_cols[agent]
                atom = ('agent',agent, State.agent_colors[agent], act_row, act_col, self.g//m)
                self.atoms.add(atom)
                if str(agent) == State.goals[act_row][act_col]:
                    atom = ('goal', agent, State.agent_colors[agent], act_row, act_col)
                    self.atoms.add(atom)
        for row in range(len(self.boxes)):
            for col in range(len(self.boxes[row])):
                if self.boxes[row][col] != '':
                    box = self.boxes[row][col]
                    # dist = self.get_goal_based_on_box(box, row, col) 
                    # atom = ('progress', box, dist//5)
                    # self.atoms.add(atom)
                    atom = ('box', box, State.box_colors[ord(box) - ord('A')], row,col, self.g//m)
                    self.atoms.add(atom)
                    if State.goals[row][col] == box:
                        atom = ('goal',box,row,col)
                        self.atoms.add(atom)
                        # self.atoms.remove(('box', box, State.box_colors[ord(box) - ord('A')], row,col, self.g//100))


        # print('aaa', self.atoms, flush=True)
        return self.atoms   


    def get_goal_based_on_box(self,box,row,col):
        dist = 0
        for (r,c),goal in State.goal_locations.items():
            if goal == box:
                dist = State.all_distances[(row,col)][r][c]
                return dist
        return dist

















    # With relaxations
    def get_expanded_states_with_relaxation(self, levels_of_relaxation):
        num_agents = len(self.agent_rows)
        
        applicable_actions = [[action for action in Action if self.is_applicable(agent, action)] for agent in range(num_agents)]
        
        expanded_states = [
            self.result_with_relaxation(joint_action)
            for joint_action in product(*applicable_actions)
                if not self.is_conflicting_with_relaxation(joint_action, levels_of_relaxation)
        ]
        State._RNG.shuffle(expanded_states)
        return expanded_states


    def result_with_relaxation(self, joint_action: '[Action, ...]') -> 'State':
        '''
        Returns the state resulting from applying joint_action in this state.
        Precondition: Joint action must be applicable and non-conflicting in this state.
        '''
        
        # Copy this state.
        copy_agent_rows = self.agent_rows[:]
        copy_agent_cols = self.agent_cols[:]
        copy_boxes = [row[:] for row in self.boxes]
        new_selections = self.selections.copy()
        boxes_locations = set(self.boxes_locations)
        
        # Apply each action.
        for agent, action in enumerate(joint_action):
            if action.type is ActionType.NoOp:
                pass
            
            elif action.type is ActionType.Move:
                copy_agent_rows[agent] += action.agent_row_delta
                copy_agent_cols[agent] += action.agent_col_delta

            elif action.type is ActionType.Push:
                copy_agent_rows[agent] += action.agent_row_delta
                copy_agent_cols[agent] += action.agent_col_delta
                box_row = copy_agent_rows[agent]
                box_col = copy_agent_cols[agent]
                box = copy_boxes[box_row][box_col]
                copy_boxes[box_row][box_col] = ''
                copy_boxes[box_row + action.box_row_delta][box_col + action.box_col_delta] = box
                
                # print('aaa',boxes_locations, flush=True)
                for (r,c,b) in boxes_locations:
                    if r == box_row and c == box_col:
                        box2 = b
                        break
                boxes_locations.remove((box_row, box_col, box2))
                boxes_locations.add((box_row + action.box_row_delta,box_col + action.box_col_delta, box2))


                if new_selections != {}:
                    (boxx, brow, bcol) = (box, box_row, box_col)
                    (new_box, new_brow, new_bcol) = (boxx, box_row + action.box_row_delta,box_col + action.box_col_delta)
                    if (boxx, brow, bcol) in new_selections:
                        (goal, grow, gcol) = new_selections[(boxx, brow, bcol)]
                        del new_selections[(boxx, brow, bcol)]
                        new_selections[(new_box, new_brow, new_bcol)] = (goal, grow, gcol)

            
            elif action.type is ActionType.Pull:
                box_row = copy_agent_rows[agent] - action.box_row_delta
                box_col = copy_agent_cols[agent] - action.box_col_delta
                box = copy_boxes[box_row][box_col]
                copy_boxes[box_row][box_col] = ''
                copy_boxes[copy_agent_rows[agent]][copy_agent_cols[agent]] = box

                
                # print('aaa',boxes_locations, flush=True)
                for (r,c,b) in boxes_locations:
                    if r == box_row and c == box_col:
                        box2 = b
                        break
                boxes_locations.remove((box_row, box_col, box2))
                boxes_locations.add((copy_agent_rows[agent],copy_agent_cols[agent], box2))

                if new_selections != {}:
                    (boxx, brow, bcol) = (box, box_row, box_col)
                    (new_box, new_brow, new_bcol) = (boxx, copy_agent_rows[agent],copy_agent_cols[agent])
                    if (boxx, brow, bcol) in new_selections:    
                        (goal, grow, gcol) = new_selections[(boxx, brow, bcol)]
                        del new_selections[(boxx, brow, bcol)]
                        new_selections[(new_box, new_brow, new_bcol)] = (goal, grow, gcol)

                copy_agent_rows[agent] += action.agent_row_delta
                copy_agent_cols[agent] += action.agent_col_delta

            
        copy_state = State(copy_agent_rows, copy_agent_cols, copy_boxes)
        
        copy_state.parent = self
        copy_state.joint_action = joint_action[:]
        copy_state.g = self.g + 1
        copy_state.atoms = copy_state.update_atoms()
        # copy_state.atoms = copy_state.update_atoms_renew_atoms()
        copy_state.fixed_agents = self.fixed_agents
        copy_state.paths = self.paths
        copy_state.selections = new_selections
        copy_state.oba_mg = self.oba_mg
        copy_state.boxes_locations = boxes_locations
        copy_state.new_atoms = self.new_atoms
        
        return copy_state
    

    def is_applicable_with_relaxation(self, agent: 'int', action: 'Action', levels_of_relaxation) -> 'bool':
        agent_row = self.agent_rows[agent]
        agent_col = self.agent_cols[agent]
        agent_color = State.agent_colors[agent]
        
        if action.type is ActionType.NoOp:
            return True
            
        elif action.type is ActionType.Move:
            destination_row = agent_row + action.agent_row_delta
            destination_col = agent_col + action.agent_col_delta
            if 'free' not in levels_of_relaxation and self.is_free(destination_row, destination_col):
                return True
            return False 
        
        elif action.type is ActionType.Push:
            destination_row_agent = agent_row + action.agent_row_delta
            destination_col_agent = agent_col + action.agent_col_delta
            box_row = destination_row_agent
            box_col = destination_col_agent
            destination_row_box = box_row + action.box_row_delta
            destination_col_box = box_col + action.box_col_delta
            box = self.box_at(box_row,box_col)
            if (box != '') and self.match_colors(agent, box) and 'free' not in levels_of_relaxation and self.is_free(destination_row_box,destination_col_box):
                    return True 
            return False
                
        elif action.type is ActionType.Pull:
            destination_row_agent = agent_row + action.agent_row_delta
            destination_col_agent = agent_col + action.agent_col_delta
            box_row = agent_row - action.box_row_delta
            box_col = agent_col - action.box_col_delta
            destination_row_box = agent_row
            destination_col_box = agent_col
            box = self.box_at(box_row,box_col)
            return (box != '') and (self.match_colors(agent,box)) and self.is_free(destination_row_agent,destination_col_agent)

    def is_conflicting_with_relaxation(self, joint_action, levels_of_relaxation):
        num_agents = len(self.agent_rows)
        destination_rows = [None for _ in range(num_agents)] 
        destination_cols = [None for _ in range(num_agents)] 
        box_rows = [None for _ in range(num_agents)] 
        box_cols = [None for _ in range(num_agents)] 
        destination_box_rows = [None for _ in range(num_agents)]
        destination_box_cols = [None for _ in range(num_agents)]
        
        for agent in range(num_agents):
            if self.agent_rows[agent] == None:
                continue
            action = joint_action[agent]
            agent_row = self.agent_rows[agent]
            agent_col = self.agent_cols[agent]
            
            if action.type is ActionType.NoOp:
                pass
             
            if action.type == ActionType.Move:
                destination_rows[agent] = agent_row + action.agent_row_delta
                destination_cols[agent] = agent_col + action.agent_col_delta

            elif action.type == ActionType.Push:
                box_row = agent_row + action.agent_row_delta
                box_col = agent_col + action.agent_col_delta
                box_rows[agent] = box_row
                box_cols[agent] = box_col
                new_box_row = box_row + action.box_row_delta
                new_box_col = box_col + action.box_col_delta
                destination_rows[agent] = box_row
                destination_cols[agent] = box_col
                destination_box_rows[agent] = new_box_row
                destination_box_cols[agent] = new_box_col

            elif action.type == ActionType.Pull:
                destination_rows[agent] = agent_row + action.agent_row_delta
                destination_cols[agent] = agent_col + action.agent_col_delta
                box_row = agent_row - action.box_row_delta  # Box moves to the agent's original position
                box_col = agent_col - action.box_col_delta
                box_rows[agent] = box_row
                box_cols[agent] = box_col
                destination_box_rows[agent] = agent_row
                destination_box_cols[agent] = agent_col

        for a1 in range(num_agents):
            if self.agent_rows[a1] == None:
                continue
            if joint_action[a1].type == ActionType.NoOp:
                continue
            
            for a2 in range(a1 + 1, num_agents):
                if joint_action[a2].type == ActionType.NoOp:
                    continue
                
                # Moving into same cell?
                if 'free' not in levels_of_relaxation and destination_rows[a1] == destination_rows[a2] and destination_cols[a1] == destination_cols[a2]:
                    return True
                #Exchanging cells
                if 'free' not in levels_of_relaxation and (destination_rows[a1] == self.agent_rows[a2] and destination_box_cols[a1] == self.agent_cols[a2]) and (destination_rows[a2] == self.agent_rows[a1] and destination_box_cols[a2] == self.agent_cols[a1]):
                    return True
                # Agent moving to destination of box
                if 'free' not in levels_of_relaxation and destination_rows[a1] == destination_box_rows[a2] and destination_cols[a1] == destination_box_cols[a2]:
                    return True
                if 'free' not in levels_of_relaxation and destination_rows[a2] == destination_box_rows[a1] and destination_cols[a2] == destination_box_cols[a1]:
                    return True
                # 2 agent moving the same box
                if 'free' not in levels_of_relaxation and box_rows[a1] == box_rows[a2] and box_cols[a1] == box_cols[a2]:
                    return True
                    
                # 2 boxes into same cell
                if 'free' not in levels_of_relaxation and destination_box_rows[a1] is not None and destination_box_rows[a2] is not None:
                    if destination_box_rows[a1] == destination_box_rows[a2] and destination_box_cols[a1] == destination_box_cols[a2]:
                        return True

        return False
    
    def delete_colors(self, color):
        boxes = []
        agents = []

        # agent, boxes, goals, 
        for row in range(len(self.boxes)):
            for col in range(len(self.boxes[row])):
                box = self.boxes[row][col]
                if box != '' and State.box_colors[ord(box)-ord('A')] == color and box not in boxes:
                    boxes.append(box)
        for agent in range(len(self.agent_rows)):
            if State.agent_colors[agent] == color:
                agents.append(agent)

        # agents = [1,3,8]
        agents = sorted(agents)
        designations = {a:'' for a in range(10)}
        i = 0
        for a in agents:
            designations[i] = a
            i+=1
        # print('designations are', designations, agents, flush=True)   #{0:1, 1:3, 2:8}
        #boxes = [C,F,G]
        boxes = sorted(boxes)
        designations_b = {b:'' for b in [chr(i) for i in range(65,91)]}   #{'A':'C', 'B':'F', 'C':'G'}
        
        i = 65
        for b in boxes:
            designations_b[chr(i)] = b
            i += 1
        designations_b_reverse = {v: k for k, v in designations_b.items() if v != ''}
        # print('designations_b are', designations_b, boxes, designations_b_reverse, flush=True)

        for row in range(len(self.boxes)):
            for col in range(len(self.boxes[row])):
                box = self.boxes[row][col]
                if box not in boxes:
                    self.boxes[row][col] = ''
                else:
                    self.boxes[row][col] = designations_b_reverse[box]
                goal = State.goals[row][col]
                if goal not in boxes:
                    State.goals[row][col] = ''
                else:
                    State.goals[row][col] = designations_b_reverse[goal]
        
        for agent in range(len(self.agent_rows)):
            if agent not in agents:
                self.agent_rows[agent] = None
                self.agent_cols[agent] = None
                State.agent_colors[agent] = None
        self.agent_rows = [a for a in self.agent_rows if a != None]
        self.agent_cols = [a for a in self.agent_cols if a != None]
        State.agent_colors = [a for a in State.agent_colors if a != None]

        for i in range(len(State.box_colors)):
            box = chr(i+65)
            if box not in boxes:
                State.box_colors[i] = None
        State.box_colors = [b for b in State.box_colors if b != None]

        return self, designations, designations_b

    def compute_paths(self, plan):
        agents_positions = {i:[] for i in range(len(self.agent_rows))}
        for agent in range(len(self.agent_rows)):
            agents_positions[agent].append((self.agent_rows[agent], self.agent_cols[agent]))
        plan.reverse()
        for joint_action in plan:
            for agent,action in enumerate(joint_action):
                last_position = agents_positions[agent][-1]
                movement = (action.agent_row_delta, action.agent_col_delta)
                new_position = (last_position[0]-movement[0], last_position[1]-movement[1])
                agents_positions[agent].append(new_position)
        
        final_agent_positions = {k:v[::-1] for k,v in agents_positions.items()}    
        return final_agent_positions

    def compute_paths_box(self, plan, box_eliminated):
        passed_box = []
        box_pos = []
        box_positions = {}
        for row in range(len(self.boxes)):
            for col in range(len(self.boxes[row])):
                if self.boxes[row][col] != '':
                    box = self.boxes[row][col]
                    box_positions[box] = [(row,col)]
                    break
        plan.reverse()
        for joint_action in plan:
            for agent,action in enumerate(joint_action):
                last_position = box_positions[box][-1]
                movement = (action.box_row_delta, action.box_col_delta)
                new_position = (last_position[0]-movement[0], last_position[1]-movement[1])
                box_positions[box].append(new_position)

                for (b,r,c) in box_eliminated:
                    if r == new_position[0] and c == new_position[1]:
                        passed_box.append((b,r,c))
        # print('box_positions', box_positions, flush=True)
        final_box_positions = {k:v[::-1] for k,v in box_positions.items()}    
        return final_box_positions, passed_box



    def relax_problem_based_on_box(self, box, brow, bcol, goal, grow,gcol):
        agents = []
        box_eliminated = []
        for row in range(len(self.boxes)):
            for col in range(len(self.boxes[row])):
                if self.boxes[row][col] != '':
                    box_eliminated.append((self.boxes[row][col],row,col))
                self.boxes[row][col] = ''
                State.goals[row][col] = ''
        
        self.boxes[brow][bcol] = 'A'
        State.goals[grow][gcol] = 'A'


        for agent in range(len(self.agent_rows)):
            if State.agent_colors[agent] == State.box_colors[ord(box)-ord('A')]:
                agents.append(agent)

        box_color = State.box_colors[ord(box)-ord('A')]
        for i in range(len(State.box_colors)):
            State.box_colors[i] = ''
        State.box_colors[0] = box_color
        



        # agents = [1,3,8]
        agents = sorted(agents)
        designations = {a:'' for a in range(10)}
        i = 0
        for a in agents:
            designations[i] = a
            i+=1
        for agent in range(len(self.agent_rows)):
            if agent not in agents:
                self.agent_rows[agent] = None
                self.agent_cols[agent] = None
                State.agent_colors[agent] = None
        self.agent_rows = [a for a in self.agent_rows if a != None]
        self.agent_cols = [a for a in self.agent_cols if a != None]
        State.agent_colors = [a for a in State.agent_colors if a != None]
        
        return self, designations, box_eliminated


    # Functions implemented by me


    ############### OBA ############### 
    def get_expanded_states_with_conflict_between_states(self, normal_state, frontier) -> '[State, ...]':
        num_agents = len(self.agent_rows)
        
        # Determine list of applicable action for each individual agent.
        applicable_actions_in_relax_state = [[action for action in Action 
                                                if self.oba_is_applicable(agent, action)] 
                                                for agent in range(num_agents)]
        applicable_actions_in_normal_state = [[action for action in Action 
                                                if self.is_applicable(agent, action)] 
                                                for agent in range(num_agents)]
        

        
        # Iterate over joint actions, check conflict and generate child states.
        expanded_states = [
                self.oba_result(joint_action)
                for joint_action in product(*applicable_actions_in_normal_state)
                    if not self.is_conflicting(joint_action)
        ]
        State._RNG.shuffle(expanded_states)
        return expanded_states


    def oba_update_atoms(self):
        self.atoms = set()
        for agent in range(len(self.agent_rows)):
            atom = ('agent',agent, State.agent_colors[agent], self.agent_rows[agent], self.agent_cols[agent])
            self.atoms.add(atom)
            if str(agent) == State.goals[self.agent_rows[agent]][self.agent_cols[agent]]:
                atom = ('goal', agent, State.agent_colors[agent], self.agent_rows[agent], self.agent_cols[agent])
                self.atoms.add(atom)        
        for row in range(len(self.boxes)):
            for col in range(len(self.boxes[row])):
                if self.boxes[row][col] != '' and (row,col,self.boxes[row][col]) in self.oba_mg: 
                    box = self.boxes[row][col]
                    atom = ('box', box, State.box_colors[ord(box) - ord('A')], row,col)
                    self.atoms.add(atom)
                    if State.goals[row][col] == box:
                        atom = ('goal',box,row,col)
                        self.atoms.add(atom)
        return self.atoms

    def oba_result(self, joint_action: '[Action, ...]') -> 'State':
        '''
        Returns the state resulting from applying joint_action in this state.
        Precondition: Joint action must be applicable and non-conflicting in this state.
        '''
        
        # Copy this state.
        copy_agent_rows = self.agent_rows[:]
        copy_agent_cols = self.agent_cols[:]
        copy_boxes = [row[:] for row in self.boxes]
        new_selections = copy.deepcopy(self.selections)
        iw = copy.deepcopy(self.iw)
        qqueue = self.qqueue[:]
        rqueue = self.rqueue[:]
        lenq = copy.deepcopy(self.lenq)
        lenr = copy.deepcopy(self.lenr)


        for agent in range(len(self.agent_rows)):
            action = joint_action[agent]
            box = None
            if action.type == ActionType.Push:
                boxrow, boxcol = self.agent_rows[agent]+action.agent_row_delta, self.agent_cols[agent]+action.agent_col_delta
                box = self.boxes[boxrow][boxcol]

            elif action.type == ActionType.Pull:
                boxrow, boxcol = self.agent_rows[agent]-action.box_row_delta, self.agent_cols[agent]-action.box_col_delta
                box = self.boxes[boxrow][boxcol]
            
            if box != None and (boxrow,boxcol,box) not in self.oba_mg:

                if (boxrow,boxcol,box) in qqueue:
                    qqueue.remove((boxrow,boxcol,box))
                    qqueue.insert(0,(boxrow,boxcol,box))

                elif (boxrow,boxcol,box) not in qqueue and len(qqueue) < lenq:
                    qqueue.insert(0,(boxrow,boxcol,box))
                    iw+=1

                elif (boxrow,boxcol,box) not in qqueue and len(qqueue) >= lenq and (boxrow,boxcol,box) not in rqueue :
                    object = qqueue.pop()
                    rqueue.insert(0,object)
                    qqueue.insert(0,(boxrow,boxcol,box))
                    if len(rqueue) > lenr:
                        obj = rqueue.pop()

                elif (boxrow,boxcol,box) not in qqueue and len(qqueue) >= lenq and (boxrow,boxcol,box) in rqueue:
                    lenq += 1
                    iw += 1
                    qqueue.insert(0,(boxrow,boxcol,box))
                    rqueue.remove((boxrow,boxcol,box))
                    

        oba_mg = set(self.oba_mg)    

        # Apply each action.
        for agent, action in enumerate(joint_action):
            if action.type is ActionType.NoOp:
                pass
            
            elif action.type is ActionType.Move:
                copy_agent_rows[agent] += action.agent_row_delta
                copy_agent_cols[agent] += action.agent_col_delta

            elif action.type is ActionType.Push:
                copy_agent_rows[agent] += action.agent_row_delta
                copy_agent_cols[agent] += action.agent_col_delta
                box_row = copy_agent_rows[agent]
                box_col = copy_agent_cols[agent]
                box = copy_boxes[box_row][box_col]
                copy_boxes[box_row][box_col] = ''
                copy_boxes[box_row + action.box_row_delta][box_col + action.box_col_delta] = box

                if (box_row, box_col, box) in oba_mg:
                    oba_mg.remove((box_row, box_col, box))
                    oba_mg.add((box_row + action.box_row_delta,box_col + action.box_col_delta, box))
                elif (box_row, box_col, box) in qqueue:
                    index = qqueue.index((box_row, box_col, box))
                    qqueue[index] = (box_row + action.box_row_delta,box_col + action.box_col_delta, box)
                elif (box_row, box_col, box) in rqueue:
                    index = rqueue.index((box_row, box_col, box))
                    rqueue[index] = (box_row + action.box_row_delta,box_col + action.box_col_delta, box)

                

                if new_selections != {}:
                    (boxx, brow, bcol) = (box, box_row, box_col)
                    (new_box, new_brow, new_bcol) = (boxx, box_row + action.box_row_delta,box_col + action.box_col_delta)
                    if (boxx, brow, bcol) in new_selections:
                        (goal, grow, gcol) = new_selections[(boxx, brow, bcol)]
                        del new_selections[(boxx, brow, bcol)]
                        new_selections[(new_box, new_brow, new_bcol)] = (goal, grow, gcol)

            
            elif action.type is ActionType.Pull:
                box_row = copy_agent_rows[agent] - action.box_row_delta
                box_col = copy_agent_cols[agent] - action.box_col_delta
                box = copy_boxes[box_row][box_col]
                copy_boxes[box_row][box_col] = ''
                copy_boxes[copy_agent_rows[agent]][copy_agent_cols[agent]] = box

                if (box_row, box_col, box) in oba_mg:
                    oba_mg.remove((box_row, box_col, box))
                    oba_mg.add((box_row + action.box_row_delta,box_col + action.box_col_delta, box))
                elif (box_row, box_col, box) in qqueue:
                    index = qqueue.index((box_row, box_col, box))
                    qqueue[index] = (box_row + action.box_row_delta,box_col + action.box_col_delta, box)
                elif (box_row, box_col, box) in rqueue:
                    index = rqueue.index((box_row, box_col, box))
                    rqueue[index] = (box_row + action.box_row_delta,box_col + action.box_col_delta, box)

                if new_selections != {}:
                    (boxx, brow, bcol) = (box, box_row, box_col)
                    (new_box, new_brow, new_bcol) = (boxx, copy_agent_rows[agent],copy_agent_cols[agent])
                    if (boxx, brow, bcol) in new_selections:    
                        (goal, grow, gcol) = new_selections[(boxx, brow, bcol)]
                        del new_selections[(boxx, brow, bcol)]
                        new_selections[(new_box, new_brow, new_bcol)] = (goal, grow, gcol)

                copy_agent_rows[agent] += action.agent_row_delta
                copy_agent_cols[agent] += action.agent_col_delta

            
        copy_state = State(copy_agent_rows, copy_agent_cols, copy_boxes)
        
        copy_state.parent = self
        copy_state.joint_action = joint_action[:]
        copy_state.g = self.g + 1
        copy_state.atoms = copy_state.update_atoms()
        copy_state.fixed_agents = self.fixed_agents
        copy_state.paths = self.paths
        copy_state.selections = new_selections
        copy_state.oba_mg = oba_mg
        copy_state.iw = iw
        copy_state.qqueue = qqueue
        copy_state.rqueue = rqueue
        copy_state.lenq = lenq
        copy_state.lenr = lenr
        
        return copy_state


    
    def oba_is_applicable(self, agent: 'int', action: 'Action') -> 'bool':
        agent_row = self.agent_rows[agent]
        agent_col = self.agent_cols[agent]
        agent_color = State.agent_colors[agent]
        
        if action.type is ActionType.NoOp:
            return True
            
        elif action.type is ActionType.Move:
            destination_row = agent_row + action.agent_row_delta
            destination_col = agent_col + action.agent_col_delta
            return self.oba_is_free(destination_row, destination_col)
        
        elif action.type is ActionType.Push:
            destination_row_agent = agent_row + action.agent_row_delta
            destination_col_agent = agent_col + action.agent_col_delta
            box_row = destination_row_agent
            box_col = destination_col_agent
            destination_row_box = box_row + action.box_row_delta
            destination_col_box = box_col + action.box_col_delta
            box = self.box_at(box_row,box_col)
            return (box != '') and (self.match_colors(agent, box)) and (box in self.oba_mg) and self.is_free(destination_row_box,destination_col_box) 
                
        elif action.type is ActionType.Pull:
            destination_row_agent = agent_row + action.agent_row_delta
            destination_col_agent = agent_col + action.agent_col_delta
            box_row = agent_row - action.box_row_delta
            box_col = agent_col - action.box_col_delta
            destination_row_box = agent_row
            destination_col_box = agent_col
            box = self.box_at(box_row,box_col)
            return (box != '') and (self.match_colors(agent,box)) and (box in self.oba_mg) and self.is_free(destination_row_agent,destination_col_agent)


    def oba_is_free(self, row: 'int', col: 'int') -> 'bool':
        return not State.walls[row][col] and (self.boxes[row][col] not in self.oba_mg) and self.agent_at(row, col) is None
    




    def goals_not_achieved_with_state(self,state):
        result = len(State.goal_locations)
        for (row,col),goal in State.goal_locations.items():
            if state.boxes[row][col] == goal:
                result -= 1 
        return result






    def goals_not_achieved(self):
        result = len(State.goal_locations)
        for (row,col),goal in State.goal_locations.items():
            if self.boxes[row][col] == goal:
                result -= 1 
        return result

    def manhattan_distance_avoiding_walls(self, state, method_focus_on_actions) -> 'int':
            result = 0
            distance_agent_box_reduced_by=1000
            if method_focus_on_actions:
                for agent,action in enumerate(state.joint_action):
                    if action.type == ActionType.NoOp:
                        continue
                    elif action.type == ActionType.Move:
                        closest_box = self.find_box_based_on_agent(state, agent, state.agent_rows[agent], state.agent_cols[agent])
                        if closest_box == None:
                            continue
                        result += closest_box[3]
                    elif action.type == ActionType.Push:
                        box_row, box_col = state.agent_rows[agent] + action.box_row_delta, state.agent_cols[agent] + action.box_col_delta
                        box = state.boxes[box_row][box_col]
                        goal_row, goal_col = list(State.goal_locations.keys())[list(State.goal_locations.values()).index(box)]
                        result += State.all_distances[(goal_row,goal_col)][box_row][box_col]
                    elif action.type == ActionType.Pull:
                        box_row, box_col = state.agent_rows[agent] - action.agent_row_delta, state.agent_cols[agent] - action.agent_col_delta
                        box = state.boxes[box_row][box_col]
                        goal_row, goal_col = list(State.goal_locations.keys())[list(State.goal_locations.values()).index(box)]
                        result += State.all_distances[(goal_row,goal_col)][box_row][box_col]
                return result
            elif not method_focus_on_actions:
                for (row,col),goal in State.goal_locations.items():

                    if '0' <= goal <= '9':
                        agent = ord(goal) - ord('0')
                        agent_row, agent_col = state.agent_rows[agent], state.agent_cols[agent]
                        result += State.all_distances[(row,col)][agent_row][agent_col]

                    elif 'A' <= goal <= 'Z':
                        if state.boxes[row][col] == goal:
                            continue
                        closest_box = self.find_box_based_on_goal(state, goal, row, col)
                        if closest_box == None:
                            continue
                        box, box_row, box_col = closest_box[1], closest_box[2], closest_box[3]
                        dist = State.all_distances[(row,col)][box_row][box_col]
                        result += dist
                if distance_agent_box_reduced_by > 0:
                    for agent in range(len(state.agent_rows)):
                        agent, agent_row, agent_col = str(agent), state.agent_rows[agent], state.agent_cols[agent]
                        box_info = self.find_box_based_on_agent(state, agent, agent_row, agent_col)
                        if box_info is not None:
                            box, box_row, box_col,distance = box_info
                            result += distance/distance_agent_box_reduced_by
                return result

    def find_box_based_on_agent(self, state, agent, arow, acol):
        boxes = []
        agent_color = State.agent_colors[ord(str(agent))-ord('0')]
        for (row,col,box) in state.boxes_locations:
            if box != '' and State.box_colors[ord(box)-ord('A')] == agent_color and State.goals[row][col] != box:
                dist = State.all_distances[(arow,acol)][row][col]
                boxes.append([box, row, col, dist])
        if boxes == []:
            return None
        return sorted(boxes, key=lambda x:x[3])[0]
    
    def precompute_distance_map(self, row, col):
        max_rows = len(self.boxes)
        max_cols = len(self.boxes[0])
        distance_map = [[float('inf')]*max_cols for _ in range(max_rows)]
        queue = deque([(row, col, 0)])
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

    def find_box_based_on_goal(self, state, goal, grow, gcol):
        boxes = []
        distance_map = State.all_distances[(grow,gcol)]
        for (row,col,box) in state.boxes_locations:
            if state.boxes[row][col] == goal and State.goals[row][col] != state.boxes[row][col]:
                bisect.insort(boxes, (distance_map[row][col],state.boxes[row][col],row,col))
        if boxes == []:
            return None
        return boxes[0]
    
    def get_goal_locations(self):
        goal_locations = {}
        for row in range(len(State.goals)):
            for col in range(len(State.goals[row])):
                if State.goals[row][col] != '':
                    goal = State.goals[row][col]
                    goal_locations[(row,col)] = goal
        return goal_locations
    
    def copy_state(self):
        agent_rows = copy.deepcopy(self.agent_rows)
        agent_cols = copy.deepcopy(self.agent_cols)
        boxes = copy.deepcopy(self.boxes)
        fixed_agent = copy.deepcopy(self.fixed_agents)
        parent = copy.deepcopy(self.parent)
        joint_action = copy.deepcopy(self.joint_action)
        g = copy.deepcopy(self.g)
        atoms = copy.deepcopy(self.atoms)
        new_atoms = copy.deepcopy(self.new_atoms)
        oba_mg = copy.deepcopy(self.oba_mg)
        boxes_locations = copy.deepcopy(self.boxes_locations)
        
        copy_state = State(agent_rows, agent_cols, boxes)
        copy_state.fixed_agents = fixed_agent
        copy_state.parent = parent
        copy_state.joint_actiot = joint_action
        copy_state.g = g
        copy_state.atoms = atoms
        copy_state.new_atoms = new_atoms
        copy_state.oba_mg = oba_mg
        copy_state.boxes_locations = boxes_locations

        return copy_state

        

    def __hash__(self):
        if self._hash is None:
            prime = 31
            _hash = 1
            _hash = _hash * prime + hash(tuple(self.agent_rows))
            _hash = _hash * prime + hash(tuple(self.agent_cols))
            _hash = _hash * prime + hash(tuple(State.agent_colors))
            _hash = _hash * prime + hash(tuple(tuple(row) for row in self.boxes))
            _hash = _hash * prime + hash(tuple(State.box_colors))
            _hash = _hash * prime + hash(tuple(tuple(row) for row in State.goals))
            _hash = _hash * prime + hash(tuple(tuple(row) for row in State.walls))
            self._hash = _hash
        return self._hash
    
    def __eq__(self, other):
        if self is other: return True
        if not isinstance(other, State): return False
        if self.agent_rows != other.agent_rows: return False
        if self.agent_cols != other.agent_cols: return False
        if State.agent_colors != other.agent_colors: return False
        if State.walls != other.walls: return False
        if self.boxes != other.boxes: return False
        if State.box_colors != other.box_colors: return False
        if State.goals != other.goals: return False
        return True
    
    def __repr__(self):
        lines = []
        for row in range(len(self.boxes)):
            line = []
            for col in range(len(self.boxes[row])):
                if self.boxes[row][col] != '': line.append(self.boxes[row][col])
                elif State.walls[row][col] is not None: line.append('+')
                elif self.agent_at(row, col) is not None: line.append(self.agent_at(row, col))
                else: line.append(' ')
            lines.append(''.join(line))
        return '\n'.join(lines)
