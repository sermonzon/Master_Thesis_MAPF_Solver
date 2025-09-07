import memory
import time
import sys
from state import State
from heuristic import Heuristic, HeuristicAlternative, Heuristicderelax
from action import Action, ActionType
import copy
from collections import defaultdict, Counter
from scipy.optimize import linear_sum_assignment



globals().update(Action.__members__)
start_time = time.perf_counter()



def search(initial_state, frontier, method):
    
    if method == 'iw' or method == 'bfws':
        
        iterations = 0
        iw = 1

        # Start the iw with the number of goals
        start_iw_with_number_of_goals = False
        if start_iw_with_number_of_goals:
            iw = sum(1 for row in State.goals for col in row if col != '')

        while True:
            if memory.get_usage() > memory.max_usage:
                print('Maximum memory usage exceeded.', file=sys.stderr, flush=True)
                return None, None, None, None, None, None, None
            frontier.empty_frontier()
            initial_state.iw = iw 
            frontier.iw = iw
            print('#Trying to find solution with iterative width', frontier.iw,  flush=True)
                
            solution, explored = search_iw(initial_state, frontier)
            if solution != None:
                print('#-------------Information about the solution-------------', flush=True)
                print('#Found solution with width',iw, flush=True)
                print('#The length of the state space is',len(frontier.state_space), flush=True)
                print('#-------------------------------------------------------', flush=True)
                return solution, iw, len(frontier.state_space), time.perf_counter()-start_time

            iw+=1

    elif method == 'siw':
        iterations = 0
        iw = 1
        goals_achieved = 0
        a = 0

        start_iw_with_number_of_goals = False
        if start_iw_with_number_of_goals:
            iw = sum(1 for row in State.goals for col in row if col != '')

        while True:
            if memory.get_usage() > memory.max_usage:
                print('Maximum memory usage exceeded.', file=sys.stderr, flush=True)
                return None, 0, None, None
            
            frontier.empty_frontier()
            frontier.iw = iw
            solution, plan, state = search_siw(initial_state, frontier, goals_achieved)
            if solution == 0:
                print('#tried to find solution with iw', iw,', now try with',iw+1, flush=True)
                iw+=1
                frontier.empty_atoms()
            elif solution == 1: #found solution
                print('#found solution with iw',iw, flush=True)
                print('#length state space', len(frontier.state_space), flush=True)
                return state.extract_plan(), 0, len(frontier.state_space),  time.perf_counter()-start_time
            elif solution == 2: #found 1 more goal, reset iw to 1
                print('#found one more goal with iw',iw , flush=True)
                iw=1
                goals_achieved+=1
                initial_state = state



    elif method == 'oba':
        print('#Starting oba', flush=True)
        initial_state.oba_mg = frontier.heuristic.get_initial_mg()
        # print('#The initial m_g is', initial_state.oba_mg, flush=True)
        state = initial_state.copy_state()
        iw = 1

        frontier.empty_frontier()
        frontier.iw = iw
        while True:
            if memory.get_usage() > memory.max_usage:
                print('Maximum memory usage exceeded.', file=sys.stderr, flush=True)
                return None
            frontier.empty_frontier()
            frontier.iw = iw
            state.iw = iw
            solution, final_iw = search_oba(initial_state, frontier)
            
            if solution != None:
                print('#--------------------------------------------------------', flush=True)
                print('#-------------Information about the solution-------------', flush=True)
                print('#Found solution with width',final_iw, flush=True)
                print('#The length of the state space is',len(frontier.state_space), flush=True)
                print('#-------------------------------------------------------', flush=True)
                return solution, final_iw, len(frontier.state_space), time.perf_counter()-start_time
            iw+=1

    elif method == 'derelaxcolor': # for color-derelaxation
        iterations = 0

        initialboxes, initialagentrows, initialagentcols = copy.deepcopy(initial_state.boxes), copy.deepcopy(initial_state.agent_rows), copy.deepcopy(initial_state.agent_cols)
        initialgoals, initialagentcolors, initialboxcolors = copy.deepcopy(State.goals), copy.deepcopy(State.agent_colors), copy.deepcopy(State.box_colors)
        colors = [c for c in list(set(State.agent_colors)) if c != None]
        solutions = {}
        for color in colors:
            # print('#Finding color', color)
            initial_state.boxes, initial_state.agent_rows, initial_state.agent_cols = copy.deepcopy(initialboxes), copy.deepcopy(initialagentrows), copy.deepcopy(initialagentcols)
            State.goals, State.agent_colors, State.box_colors = copy.deepcopy(initialgoals), copy.deepcopy(initialagentcolors), copy.deepcopy(initialboxcolors)
            iw = 1
            found = False
            while not found:
                if memory.get_usage() > memory.max_usage:
                    print('Maximum memory usage exceeded.', file=sys.stderr, flush=True)
                    return None
                frontier.empty_frontier()
                frontier.iw = iw
                # print('#try to find solution with IW =',iw, flush=True)
                solution, paths = search_derelax(initial_state, frontier, color)
                if solution != None:
                    # print('#found solution with width',iw)
                    solutions[color] = solution
                    found = True
                    for p in paths.keys():
                        frontier.paths[p] = paths[p]

                iw+=1
        # print('solutions are', solutions, flush=True)
        print('#-------------Found all paths, now solve the problem-------------', flush=True)
        frontier.all_paths_found = True
        initial_state.boxes, initial_state.agent_rows, initial_state.agent_cols = copy.deepcopy(initialboxes), copy.deepcopy(initialagentrows), copy.deepcopy(initialagentcols)
        State.goals, State.agent_colors, State.box_colors = copy.deepcopy(initialgoals), copy.deepcopy(initialagentcolors), copy.deepcopy(initialboxcolors)
        frontier.heuristic.compute_goals(initial_state)

        iterations = 0

        iw = 1
        while True:
            if memory.get_usage() > memory.max_usage:
                print('Maximum memory usage exceeded.', file=sys.stderr, flush=True)
                return None
            frontier.empty_frontier()
            frontier.iw = iw
            print('try to find solution with iterative width',iw, flush=True)
            solution,_ = search_iw(initial_state, frontier)
            if solution != None:
                print('found with width',iw)
                return solution, iw, len(frontier.state_space), time.perf_counter()-start_time
            iw+=1

    elif method == 'derelaxbox':  # for box-relaxation
        iterations = 0
        initialboxes, initialagentrows, initialagentcols = copy.deepcopy(initial_state.boxes), copy.deepcopy(initial_state.agent_rows), copy.deepcopy(initial_state.agent_cols)
        initialgoals, initialagentcolors, initialboxcolors = copy.deepcopy(State.goals), copy.deepcopy(State.agent_colors), copy.deepcopy(State.box_colors)
        solutions = {}
        passed_boxes = {}
        boxes = Heuristicderelax.get_boxes(Heuristicderelax, initial_state)
        goals = Heuristicderelax.get_goals(Heuristicderelax, initial_state)
        selections = {}
        unique_boxes = set([b[0] for b in boxes])
        boxes_with_goals = set()
        for unique_box in unique_boxes:
            boxes2 = [b for b in boxes if b[0] == unique_box]
            goals2 = [g for g in goals if g[0] == unique_box]
            boxes_pos = [(a[2], a[3]) for a in boxes if a[0] == unique_box]
            goals_pos = [(a[1], a[2]) for a in goals if a[0] == unique_box]
            frontier.heuristic.compute_goals(initial_state)
            cost = [[None for _ in range(len(goals_pos))] for _ in range(len(boxes_pos))]

            
            for i in range(len(boxes_pos)):
                for j in range(len(goals_pos)):
                    cost[i][j] = frontier.heuristic.all_distances[(boxes_pos[i][0], boxes_pos[i][1])][goals_pos[j][0]][goals_pos[j][1]]
            row_ind, col_ind = linear_sum_assignment(cost)
            goal_assignment = {
                box_pos: goal_pos
                for box_pos, goal_pos in zip(row_ind, col_ind)
            }            
            
            for i in range(len(boxes2)):
                if i not in goal_assignment:
                    continue
                boxes_with_goals.add((boxes2[i]))
                sel = goal_assignment[i]
                selections[(boxes2[i][0], boxes2[i][2], boxes2[i][3])] = (goals2[sel][0], goals2[sel][1], goals2[sel][2])



        print('#The selections are', selections, flush=True)
        random_selection=False
        if random_selection:
            selections = frontier.heuristic.goal_closest_to_goal(initial_state)

        initial_state.selections = selections
        for (box, color_box, brow, bcol) in boxes_with_goals:
            
            (goal, grow, gcol) = selections[(box, brow, bcol)]
            initial_state.selections = {('A', brow, bcol):('A', grow, gcol)}
            solution, paths, passed_box = search_based_on_box_goal(frontier, box, brow, bcol, goal,grow,gcol, initial_state,initialboxes, initialagentrows, initialagentcols,initialgoals, initialagentcolors, initialboxcolors)
            solutions[(box,brow,bcol)] = solution
            found = True
            passed_boxes[(box,brow,bcol)] = passed_box
            for p in paths.keys():
                initial_state.paths[p] = paths[p]

        initial_state.selections = selections 
        blocking_counts = {}
        for info_box, path in passed_boxes.items():
            (box,brow,bcol) = info_box
            for step in path:
                blocking_box = step
                if blocking_box != info_box:
                    if step not in blocking_counts:
                        blocking_counts[step] = 0
                    blocking_counts[step] += 1
        for box in passed_boxes.keys():
            if box not in blocking_counts.keys():
                blocking_counts[box] = 0
        priority_order = sorted(passed_boxes.keys(), key=lambda x: -blocking_counts[x])


        print('#-------------Found all paths, now solve the problem-------------', flush=True)
        frontier.all_paths_found = True
        frontier.priority = priority_order
        initial_state.boxes, initial_state.agent_rows, initial_state.agent_cols = copy.deepcopy(initialboxes), copy.deepcopy(initialagentrows), copy.deepcopy(initialagentcols)
        State.goals, State.agent_colors, State.box_colors = copy.deepcopy(initialgoals), copy.deepcopy(initialagentcolors), copy.deepcopy(initialboxcolors)
        frontier.heuristic.compute_goals(initial_state)

        iterations = 0

        iw = 1
        while True:
            if memory.get_usage() > memory.max_usage:
                print('Maximum memory usage exceeded.', file=sys.stderr, flush=True)
                return None
            frontier.empty_frontier()
            frontier.iw = iw
            print('try to find solution with iterative width',iw, flush=True)
            solution = search_derelax22(initial_state, frontier)
            if solution != None:
                print('#-------------Information about the solution-------------', flush=True)
                print('#Found solution with width',iw, flush=True)
                print('#The length of the state space is',len(frontier.state_space), flush=True)
                print('#Number of states taken from reserve', frontier.has_taken_from_reserve, flush=True) if frontier.has_taken_from_reserve >0 else None
                print('#-------------------------------------------------------', flush=True)
                return  solution, iw, len(frontier.state_space), time.perf_counter()-start_time
            iw+=1


        

    else: 
        print('#Other method', flush=True)
    
        iterations = 0

        frontier.add(initial_state)
        explored = set()

        while True:

            iterations += 1

            if memory.get_usage() > memory.max_usage:
                print_search_status(explored, frontier)
                print('Maximum memory usage exceeded.', file=sys.stderr, flush=True)
                return None, 0, None, None, None, None

            # Your code here...
            if frontier.is_empty():
                return None, 0, len(frontier.state_space), time.perf_counter()-start_time
            state = frontier.pop()
            if state.is_goal_state():
                return state.extract_plan(), 0, len(frontier.state_space), time.perf_counter()-start_time

            explored.add(state)
            for child in state.get_expanded_states():
                if (child not in explored) and (not frontier.contains(child)):
                    frontier.add(child)

def search_derelax(initial_state, frontier, color):
    iterations = 0
    initial_state, designations_agents, designations_boxes = initial_state.delete_colors(color)
    frontier.heuristic.compute_goals(initial_state)

    frontier.add(initial_state)
    explored = set()
    
    
    while True:
        if frontier.is_empty(): 
            return None, None

        state = frontier.pop()
        if state.is_goal_state():
            plan = state.extract_plan()
            paths = state.compute_paths(plan)
            paths = {designations_agents[k]:v for k,v in paths.items()}
            return plan, paths

        explored.add(state)
        for child in state.get_expanded_states_with_relaxation(['']):
            if (child not in explored) and (not frontier.contains(child)):
                frontier.add(child)
        
        iterations += 1

def search_derelax2(initial_state, frontier, box, brow, bcol, goal, grow, gcol):
    iterations = 0
    
    initial_state, designations_agents, box_eliminated = initial_state.relax_problem_based_on_box(box, brow, bcol, goal,grow,gcol)
    
    frontier.heuristic.compute_goals(initial_state)
    frontier.add(initial_state)
    explored = set()
    passed_box = []
    
    while True:
        if frontier.is_empty(): 
            return None, None, None
        state = frontier.pop()

        if frontier.heuristic.goals_achieved(state) == 1:
            plan = state.extract_plan()
            paths, passed_box = state.compute_paths_box(plan, box_eliminated)
            paths = {box:v for k,v in paths.items()}
            return plan, paths, passed_box

        explored.add(state)
        for child in state.get_expanded_states_with_relaxation(['']):
            if (child not in explored) and (not frontier.contains(child)):
                frontier.add(child)
        
        iterations += 1

def search_iw(initial_state, frontier):
    iterations = 0
    frontier.add(initial_state)
    explored = set()

    while True:
        if frontier.is_empty():  
            return None, explored
        state = frontier.pop()
        
        update_distances = False
        if update_distances:
            if state.parent != None and frontier.heuristic.goals_achieved(state) != frontier.heuristic.goals_achieved(state.parent):
                for row in range(len(State.goals)):
                    for col in range(len(State.goals[row])):
                        frontier.heuristic.all_distances[(row,col)] = frontier.heuristic.precompute_distance_map_with_goals_as_walls(state, None, row,col)

        if state.is_goal_state():
            return state.extract_plan(), explored
        
        explored.add(state)
        for child in state.get_expanded_states():
            if (child not in explored) and (not frontier.contains(child)):
                frontier.add(child)

        iterations += 1

def search_siw(initial_state, frontier, goals_achieved):
    iterations = 0
    frontier.add(initial_state)
    explored = set()
    while True:
        iterations+=1
        if frontier.is_empty():  #No more states to explore
            return 0, None, None
        state = frontier.pop()
        if state.is_goal_state():  #Found a global solution
            return 1, state.extract_plan(), state
        new_goals_achieved = state.goals_achieved()
        if new_goals_achieved > goals_achieved:  #Found 1 more goal
            return 2, state.extract_plan(), state
        explored.add(state)
        for child in state.get_expanded_states():
            if (child not in explored) and (not frontier.contains(child)):
                frontier.add(child)

def search_oba(initial_state, frontier, method = 'oba'):
    # print('#Searching with OBA and IW = ', frontier.iw, flush=True)
    iterations = 0
    frontier.add(initial_state)
    explored = set()

    while True:
        iterations+=1
        if frontier.is_empty():  
            return None, None
        state = frontier.pop()
        if state.is_goal_state():
            return state.extract_plan(), frontier.iw
        explored.add(state)
        for child in state.get_expanded_states_with_conflict_between_states(state, frontier):
            if (child not in explored) and (not frontier.contains(child)):
                frontier.add(child)

def print_search_status(explored, frontier):
    status_template = '#Expanded: {:8,}, #Frontier: {:8,}, #Generated: {:8,}, Time: {:3.3f} s\n[Alloc: {:4.2f} MB, MaxAlloc: {:4.2f} MB]'
    elapsed_time = time.perf_counter() - start_time
    print(status_template.format(len(explored), frontier.size(), len(explored) + frontier.size(), elapsed_time, memory.get_usage(), memory.max_usage), file=sys.stderr, flush=True)

def search_based_on_box_goal(frontier, box, brow, bcol, goal, grow, gcol, initial_state,initialboxes, initialagentrows, initialagentcols,initialgoals, initialagentcolors, initialboxcolors):
    # print('#Finding solution for box', (box, brow, bcol))
    initial_state.boxes, initial_state.agent_rows, initial_state.agent_cols = copy.deepcopy(initialboxes), copy.deepcopy(initialagentrows), copy.deepcopy(initialagentcols)
    State.goals, State.agent_colors, State.box_colors = copy.deepcopy(initialgoals), copy.deepcopy(initialagentcolors), copy.deepcopy(initialboxcolors)
    iw = 1
    found = False
    while not found:
        if memory.get_usage() > memory.max_usage:
            print('Maximum memory usage exceeded.', file=sys.stderr, flush=True)
            return None
        frontier.empty_frontier()
        frontier.iw = iw
        # print('#try to find solution with IW =',iw, flush=True)
        solution, paths, passed_box = search_derelax2(initial_state, frontier, box, brow, bcol, goal, grow,gcol)
        if solution != None:
            # print('#found solution with width',iw)
            return solution, paths, passed_box
        iw+=1

def search_derelax22(initial_state, frontier):
    iterations = 0
    frontier.add(initial_state)
    explored = set()

    while True:
        if frontier.is_empty():  
            return None
        state = frontier.pop()

        if False: #update all_distances
            if state.parent != None and frontier.heuristic.goals_achieved(state) != frontier.heuristic.goals_achieved(state.parent):
                for row in range(len(State.goals)):
                    for col in range(len(State.goals[row])):
                        frontier.heuristic.all_distances[(row,col)] = frontier.heuristic.precompute_distance_map_with_goals_as_walls(state, None, row,col)

        if state.is_goal_state():
            return state.extract_plan()
        explored.add(state)
        for child in state.get_expanded_states():
            if (child not in explored) and (not frontier.contains(child)):
                frontier.add(child)

        iterations += 1
