from abc import ABCMeta, abstractmethod
from collections import deque
import heapq
from itertools import combinations
from state import State
from heuristic import Heuristic, Heuristicderelax, HeuristicAlternative
from sortedcontainers import SortedList


# Template function
class Frontier(metaclass=ABCMeta):
    @abstractmethod
    def add(self, state: 'State'): raise NotImplementedError
    
    @abstractmethod
    def pop(self) -> 'State': raise NotImplementedError
    
    @abstractmethod
    def is_empty(self) -> 'bool': raise NotImplementedError
    
    @abstractmethod
    def size(self) -> 'int': raise NotImplementedError
    
    @abstractmethod
    def contains(self, state: 'State') -> 'bool': raise NotImplementedError
    
    @abstractmethod
    def get_name(self): raise NotImplementedError

# BFS Frontier
class FrontierBFS(Frontier):
    def __init__(self):
        super().__init__()
        self.queue = deque()
        self.set = set()
        self.state_space = set()
    
    def add(self, state: 'State'):
        self.queue.append(state)
        self.set.add(state)
        self.state_space.add(state)
    
    def pop(self) -> 'State':
        state = self.queue.popleft()
        self.set.remove(state)
        return state
    
    def is_empty(self) -> 'bool':
        return len(self.queue) == 0
    
    def size(self) -> 'int':
        return len(self.queue)
    
    def contains(self, state: 'State') -> 'bool':
        return state in self.set
    
    def get_name(self):
        return 'breadth-first search'

#DFS Frontier
class FrontierDFS(Frontier):
    def __init__(self):
        super().__init__()
        self.queue = deque()
        self.set = set()
        self.state_space = set()

    def add(self, state: 'State'):
        self.queue.appendleft(state)
        self.set.add(state)
        self.state_space.add(state)

    def pop(self) -> 'State':
        state = self.queue.popleft()
        self.set.remove(state)
        return state
        
    def is_empty(self) -> 'bool':
        return len(self.queue) == 0
    
    def size(self) -> 'int':
        return len(self.queue)
    
    def contains(self, state: 'State') -> 'bool':
        return state in self.set
    
    def get_name(self):
        return 'depth-first search'

# Best-First Frontier
class FrontierBestFirst(Frontier):
    def __init__(self, heuristic: 'Heuristic'):
        super().__init__()
        self.heuristic = heuristic
        self.priotity_queue = []
        self.set = set()
        self.counter = 0

    def add(self, state: 'State'):
        heapq.heappush(self.priotity_queue, (self.heuristic.f(state), state.g, self.counter, state))
        self.set.add(state)
        self.counter+=1

    def pop(self) -> 'State':
        state = heapq.heappop(self.priotity_queue)[-1]
        self.set.remove(state)
        return state

    def is_empty(self) -> 'bool':
        return len(self.priotity_queue) == 0
    
    def size(self) -> 'int':
        return len(self.priotity_queue)
    
    def contains(self, state: 'State') -> 'bool':
        return state in self.set
    
    def get_name(self):
        return 'best-first search using {}'.format(self.heuristic)
    
# IW Frontier
class FrontierIW(Frontier):
    def __init__(self, heuristic: 'Heuristic'):
        super().__init__()
        self.heuristic = heuristic
        self.priotity_queue = []
        self.set = set()
        self.counter = 0
        self.atoms_explored = set()
        self.iw = 1
        self.state_space = set() #alternative heuristics
        

    def add(self, state: 'State'):
        atoms = state.atoms
        new_atoms = self.new_atoms(state)
        heuristics_to_order = [self.heuristic.goals_not_achieved] 
        h_to_order = tuple(h(state) for h in heuristics_to_order)
            
        if new_atoms:
            heapq.heappush(self.priotity_queue, (
                                                self.heuristic.goals_not_achieved(state), 
                                                self.counter, state))
            self.set.add(state)
            self.state_space.add((h_to_order, 
                                  self.counter, state)) 
            self.atoms_explored.update(combinations(atoms,self.iw))
        self.counter+=1

    def pop(self) -> 'State':
        state = heapq.heappop(self.priotity_queue)[-1]
        self.set.remove(state)
        return state

    def new_atoms(self,state:'State'):
        atoms = list(combinations(state.atoms, self.iw))
        for atom in atoms:
            if atom not in self.atoms_explored:
                return True
        return False
    
    def empty_frontier(self):
        self.priotity_queue = []
        self.set = set()
        self.atoms_explored = set()
        self.counter = 0

    def is_empty(self) -> 'bool':
        return len(self.priotity_queue) == 0
    
    def size(self) -> 'int':
        return len(self.priotity_queue)
    
    def contains(self, state: 'State') -> 'bool':
        return state in self.set
    
    def get_name(self):
        return 'IW {}'.format(self.heuristic)
    
# SIW Frontier
class FrontierSIW(Frontier):
    def __init__(self, heuristic: 'Heuristic'):
        super().__init__()
        self.heuristic = heuristic
        self.priotity_queue = []
        self.set = set()
        self.counter = 0
        self.atoms_explored = set()
        self.iw = 1
        self.state_space = set()

    def add(self, state: 'State'):
        atoms = state.atoms
        new_atoms = self.new_atoms(state)
        if new_atoms:
            heapq.heappush(self.priotity_queue, (
                                                self.heuristic.goals_not_achieved(state), 
                                                self.counter, state))
            self.set.add(state)
            self.state_space.add(state)
            self.atoms_explored.update(combinations(atoms,self.iw))
        self.counter+=1

    def pop(self) -> 'State':
        state = heapq.heappop(self.priotity_queue)[-1]
        self.set.remove(state)
        return state

    def new_atoms(self,state:'State'):
        atoms = list(combinations(state.atoms, self.iw))
        for atom in atoms:
            if atom not in self.atoms_explored:
                return True
        return False
    
    def empty_frontier(self):
        self.priotity_queue = []
        self.set = set()
        self.atoms_explored = set()
        self.counter = 0

    def empty_atoms(self):
        pass
        
        
    def is_empty(self) -> 'bool':
        return len(self.priotity_queue) == 0
    
    def size(self) -> 'int':
        return len(self.priotity_queue)
    
    def contains(self, state: 'State') -> 'bool':
        return state in self.set
    
    def get_name(self):
        return 'SIW {}'.format(self.heuristic)

# BFWS and IW-H Frontier
    # Inside __init__, select self.use_bfws = True for BFWS, select self.use_bfws = False for IW-H
    # Inside __init__, select self.reintroduce_pruned_states = True for RPS
    # Inside add, Uncomment the lines for OCNA
class Frontierbfws(Frontier):
    def __init__(self, heuristic: 'Heuristic'):
        super().__init__()
        self.heuristic = heuristic
        self.priority_queue = []
        self.pruned_priority_queue = []
        self.set = set()
        self.counter = 0
        self.atoms_explored = {}
        self.iw = 1
        self.state_space = set()
        self.atoms_and_m = {}
        self.atoms_and_g = {}
        self.states_and_f = {}
        self.has_taken_from_reserve = False
        self.goals_seen = set()
        self.reintroduce_pruned_states = False # Change to True for RPS 
        self.use_bfws = False # Select True for BFWS, False for IW-H

    def add(self, state: 'State'):

        heuristics = [self.heuristic.which_goals_are_achieved]
        atoms = state.atoms
        new_atoms = self.new_atoms(state, heuristics)
        heuristics_to_order = [self.heuristic.goals_not_achieved]

        if new_atoms > 0:
            h_to_order = tuple(h(state) for h in heuristics_to_order)
            heapq.heappush(self.priority_queue, (
                                                # Uncomment for OCNA
                                                # -new_atoms,  
                                                h_to_order,
                                                self.counter, state))
            # Uncomment for OCNA 
            # state.new_atoms = new_atoms + state.parent.new_atoms if state.parent != None else new_atoms 
            self.set.add(state)
            self.state_space.add(state)
            h_state = [h(state) for h in heuristics]
            for at in combinations(atoms,self.iw):

                if at not in self.atoms_explored:
                    self.atoms_explored[at] = []
                if h_state not in self.atoms_explored[at]:
                    self.atoms_explored[at].append(h_state)

        elif self.reintroduce_pruned_states and state.parent in self.state_space:
            heapq.heappush(self.pruned_priority_queue, (
                                                # self.heuristic.which_goals_are_achieved(state),
                                                # self.states_and_f[state.parent],
                                                # self.heuristic.f(state),
                                                self.heuristic.goals_not_achieved(state),
                                                # self.heuristic.penalization_based_on_goals(state),                                    
                                                # self.heuristic.manhattan_distance_avoiding_walls(state),
                                                # -self.heuristic.get_new_atoms(state),
                                                # self.heuristic.get_new_goals_atoms(state),
                                                self.counter, state))
            self.set.add(state)
        self.counter+=1


    def pop(self) -> 'State':
        if len(self.priority_queue) != 0:
            state = heapq.heappop(self.priority_queue)[-1]
            self.set.remove(state)
        elif self.reintroduce_pruned_states:
            self.has_taken_from_reserve = True
            state = heapq.heappop(self.pruned_priority_queue)[-1]
            self.set.remove(state)
        return state
        
    def new_atoms(self,state:'State', heuristics):
        new_atoms = 0
        h_state = [h(state) for h in heuristics]

        for atom in combinations(state.atoms, self.iw):
            
            if self.use_bfws:
                if atom not in self.atoms_explored or (atom in self.atoms_explored and h_state not in self.atoms_explored[atom]) :
                    new_atoms += 1
            else:
                if atom not in self.atoms_explored or (atom in self.atoms_explored and all(len(h_state[0]) > len(a[0]) for a in self.atoms_explored[atom])):
                    new_atoms += 1         
        return new_atoms
        
       

    def empty_frontier(self):
        self.priority_queue = []
        self.set = set()
        self.atoms_explored = {}
        self.counter = 0
        self.state_space = set()
        self.has_taken_from_reserve = False

    def is_empty(self) -> 'bool':
        if self.reintroduce_pruned_states:
            return len(self.priority_queue) == 0 and len(self.pruned_priority_queue) == 0
        else:
            return len(self.priority_queue) == 0
    
    def size(self) -> 'int':
        return len(self.priority_queue)
    
    def contains(self, state: 'State') -> 'bool':
        return state in self.set
    
    def get_name(self):
        return 'BFS5 BFS6 {}'.format(self.heuristic)
    
# IW+ and IW++ Frontier
    # Inside __init__, select self.use_iw_plus = True for IW+, and self.use_iw_plusc= False for IW++
class FrontierIWplus(Frontier):
    def __init__(self, heuristic: 'Heuristic'):
        super().__init__()
        self.heuristic = heuristic
        self.priotity_queue = []
        self.set = set()
        self.counter = 0
        self.atoms_explored = set()
        self.atoms_and_new_atoms = {}
        self.iw = 1
        self.state_space = set() 
        self.use_iw_plus = True # Select True for IW+, select False for IW++

    def add(self, state: 'State'):
        atoms = state.atoms
        new_atoms = self.new_atoms(state)
        if new_atoms > 0:
            heapq.heappush(self.priotity_queue, (
                                                self.heuristic.goals_not_achieved(state), 
                                                self.counter, state))
            self.set.add(state)
            self.state_space.add((self.heuristic.f(state),state))
            self.atoms_explored.update(combinations(atoms,self.iw))
            state.new_atoms = state.parent.new_atoms + new_atoms if state.parent != None else new_atoms 
            for at in combinations(atoms,self.iw):
                if at not in self.atoms_and_new_atoms:
                    self.atoms_and_new_atoms[at] = []
                if self.use_iw_plus:
                    if state.new_atoms not in self.atoms_and_new_atoms[at]:
                        self.atoms_and_new_atoms[at].append(state.new_atoms)
                else: 
                    if (state.new_atoms,state.g) not in self.atoms_and_new_atoms[at]:
                        self.atoms_and_new_atoms[at].append((state.new_atoms,state.g))
        self.counter+=1

    def pop(self) -> 'State':
        state = heapq.heappop(self.priotity_queue)[-1]
        self.set.remove(state)
        return state

    def new_atoms(self,state:'State'):
        atoms = list(combinations(state.atoms, self.iw))
        new_at = 0
        for atom in atoms:
            if self.use_iw_plus:
                if atom not in self.atoms_explored or (atom in self.atoms_explored and state.new_atoms not in self.atoms_and_new_atoms[atom]) :
                    new_at += 1
            else: 
                if atom not in self.atoms_explored or (atom in self.atoms_explored and (state.new_atoms,state.g) not in self.atoms_and_new_atoms[atom]) :
                    new_at += 1
        return new_at
    
    def empty_frontier(self):
        self.priotity_queue = []
        self.set = set()
        self.atoms_explored = set()
        self.counter = 0
        self.atoms_and_new_atoms = {}
        self.state_space = set() 

    def is_empty(self) -> 'bool':
        return len(self.priotity_queue) == 0
    
    def size(self) -> 'int':
        return len(self.priotity_queue)
    
    def contains(self, state: 'State') -> 'bool':
        return state in self.set
    
    def get_name(self):
        return 'IW+ {}'.format(self.heuristic)

#OBA Frontier
class FrontierOBA(Frontier):
    def __init__(self, heuristic: 'Heuristic'):
        super().__init__()
        self.heuristic = heuristic
        self.priotity_queue = []
        self.set = set()
        self.counter = 0
        self.atoms_explored = {}
        self.iw = 1
        self.state_space = set() 
        self.m_g = set()
        self.reintroduced_pruned_states = False
        self.reserve_states = []

    def add(self, state: 'State'):
        heuristics = [self.heuristic.which_goals_are_achieved]
        
        atoms = state.atoms
        new_atoms = self.new_atoms(state, heuristics)
        heuristics_to_order = [self.heuristic.goals_not_achieved]
        if new_atoms or state.parent is None:
            h_to_order = tuple(h(state) for h in heuristics_to_order)
            heapq.heappush(self.priotity_queue, (
                h_to_order,
                self.counter, 
                state))
            self.set.add(state)
            self.state_space.add(state)

            h_state = [h(state) for h in heuristics] 
            for at in combinations(atoms, state.iw):
                if all( ord('0')<=ord(str(a[1]))<=ord('9') or a[0] == 'goal' or (a[3],a[4],a[1]) in state.oba_mg or (a[3],a[4],a[1]) in state.qqueue for a in at ) :
                    if at not in self.atoms_explored:
                        self.atoms_explored[at] = []
                    if h_state not in self.atoms_explored[at]:
                        self.atoms_explored[at].append(h_state)
    
        elif self.reintroduced_pruned_states and state.parent in self.state_space:
            heapq.heappush(self.reserve_states, (
                            self.heuristic.goals_not_achieved(state), 
                            self.counter, 
                            state))
            self.set.add(state)

        self.counter+=1



    def pop(self) -> 'State':
        if len(self.priotity_queue) != 0:
            state = heapq.heappop(self.priotity_queue)[-1]
            self.set.remove(state)
        elif self.reintroduced_pruned_states:
            state = heapq.heappop(self.reserve_states)[-1]
            self.set.remove(state)
        return state
    
    def get_goals(self):
        return self.heuristic.get_goals()
    
    '''
    def removable_objects(self, m_g, state):
        goals = self.get_goals()
        removable_objects = []

        ######## For now, do not remove any agents. Maybe change it later
        # for agent in range(len(state.agent_rows)):
        #     if str(agent) not in goals and agent not in m_g:
        #         removable_objects.append(['a',agent,state.agent_rows[agent],state.agent_cols[agent]])
        for row in range(len(state.boxes)):
            for col in range(len(state.boxes[row])):
                if state.boxes[row][col] != '' and state.boxes[row][col] not in goals and state.boxes[row][col] not in m_g:
                    removable_objects.append(['box',state.boxes[row][col],row,col])
        return removable_objects
    '''

    def new_atoms(self, state:'State', heuristics):
        atoms = list(combinations(state.atoms, state.iw))
        h_state = [h(state) for h in heuristics]
        for atom in atoms:

            if atom not in self.atoms_explored and \
                all(ord('0')<=ord(str(a[1]))<=ord('9') or a[0] == 'goal' or (a[3],a[4],a[1]) in state.oba_mg or (a[3],a[4],a[1]) in state.qqueue for a in atom):
                return True
            # if (atom in self.atoms_explored and h_state not in self.atoms_explored[atom]):
                # return True
            if atom in self.atoms_explored and all(len(h_state[0]) > len(a[0]) for a in self.atoms_explored[atom]):
                return True
            
        return False
    
    
    
    def empty_frontier(self):
        self.priotity_queue = []
        self.set = set()
        self.atoms_explored = {}
        self.counter = 0
        self.state_space = set() 

    def is_empty(self) -> 'bool':
        if self.reintroduced_pruned_states:
            return len(self.priotity_queue) == 0 and len(self.reserve_states) == 0
        else:
            return len(self.priotity_queue) == 0
        
    def size(self) -> 'int':
        return len(self.priotity_queue)
    
    def contains(self, state: 'State') -> 'bool':
        return state in self.set
    
    def get_name(self):
        return 'OBA {}'.format(self.heuristic)
    
# DerelaxCOLOR and DerelaxBOX Frontier
class Frontierderelax(Frontier):
    def __init__(self, heuristic: 'Heuristicderelax'):
        super().__init__()
        self.heuristic = heuristic
        self.priority_queue = []
        self.pruned_priority_queue = []
        self.set = set()
        self.counter = 0
        self.atoms_explored = {}
        self.iw = 1
        self.state_space = set()
        self.paths = {}
        self.all_paths_found = False
        self.priority = []
        self.has_taken_from_reserve = 0
        self.reintroduce_pruned_states = False
        

    def add(self, state: 'State'):
        heuristics = [self.heuristic.which_goals_are_achieved]
        state.atoms = state.update_atoms()
        atoms = state.atoms        
        new_atoms = self.new_atoms(state, heuristics)

        if new_atoms:
            if not self.all_paths_found:
                heapq.heappush(self.priority_queue, (
                                                self.heuristic.goals_not_achieved(state),
                                                self.heuristic.manhattan_distance_avoiding_walls(state),
                                                self.counter, state))
            else:
                heapq.heappush(self.priority_queue, (
                                        self.heuristic.goals_not_achieved(state),
                                        # Derelax - COLOR
                                        # self.heuristic.distance_to_path(state, self.paths),
                                        # Derelax - BOX
                                        self.heuristic.distance_to_path_boxes(state, state.paths, self.priority),
                                        self.counter, state))


            
            self.set.add(state)
            self.state_space.add(state)
            h_state = [h(state) for h in heuristics]
            for at in combinations(atoms,self.iw):
                if at not in self.atoms_explored:
                    self.atoms_explored[at] = []
                if h_state not in self.atoms_explored[at]:
                    self.atoms_explored[at].append(h_state)

        elif self.reintroduce_pruned_states and state.parent in self.state_space and self.all_paths_found:
            heapq.heappush(self.pruned_priority_queue, (
                                                self.heuristic.goals_not_achieved(state),
                                                # self.heuristic.distance_box_to_selected_goals(state),
                                                self.heuristic.distance_to_path(state, self.paths),#, self.priority),
                                                # self.heuristic.which_goals_are_achieved(state),
                                                # self.states_and_f[state.parent],
                                                # self.heuristic.f(state),
                                                # self.heuristic.goals_not_achieved(state),
                                                # self.heuristic.penalization_based_on_goals(state),                                    
                                                # self.heuristic.manhattan_distance_avoiding_walls(state),
                                                # -self.heuristic.get_new_atoms(state),
                                                # self.heuristic.get_new_goals_atoms(state),
                                                self.counter, state))
            self.set.add(state)

        self.counter+=1





    def pop(self) -> 'State':
        if len(self.priority_queue) != 0:
            state = heapq.heappop(self.priority_queue)[-1]
            self.set.remove(state)
        elif self.reintroduce_pruned_states:
            state = heapq.heappop(self.pruned_priority_queue)[-1]
            self.set.remove(state)
            self.has_taken_from_reserve += 1
        return state
    
    
    def new_atoms(self,state, heuristics):
            h_state = [h(state) for h in heuristics]
            for atom in combinations(state.atoms, self.iw):
                if atom not in self.atoms_explored or (atom in self.atoms_explored and all(len(h_state[0]) > len(a[0]) for a in self.atoms_explored[atom])):
                    return True        
            return False


    def empty_frontier(self):
        self.priority_queue = []
        self.set = set()
        self.atoms_explored = {}
        self.counter = 0
        self.state_space = set()
        self.pruned_priority_queue = []
        self.atoms_explored = {}
        self.state_space = set()
        
    def is_empty(self) -> 'bool':
        if self.reintroduce_pruned_states:
            return len(self.priority_queue) == 0 and len(self.pruned_priority_queue) == 0
        else:
            return len(self.priority_queue) == 0
    
    def size(self) -> 'int':
        return len(self.priority_queue)
    
    def contains(self, state: 'State') -> 'bool':
        return state in self.set
    
    def get_name(self):
        return 'Relaxation {}'.format(self.heuristic)
    