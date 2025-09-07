import os


import argparse
import io
import sys
import time

import cProfile

import memory
from color import Color
from state import State
from frontier import * 

from heuristic import * 
from graphsearch import search

from openpyxl import load_workbook

# start_time = time.perf_counter()
class SearchClient:
    @staticmethod
    def parse_level(server_messages) -> 'State':
        # We can assume that the level file is conforming to specification, since the server verifies this.
        # Read domain.
        server_messages.readline() # #domain
        server_messages.readline() # hospital
        
        # Read Level name.
        server_messages.readline() # #levelname
        levelname = server_messages.readline() # <name>
        levelname = levelname[:len(levelname)-1]
        # Read colors.
        server_messages.readline() # #colors
        agent_colors = [None for _ in range(10)]
        box_colors = [None for _ in range(26)]
        line = server_messages.readline()
        while not line.startswith('#'):
            split = line.split(':')
            color = Color.from_string(split[0].strip())
            entities = [e.strip() for e in split[1].split(',')]
            for e in entities:
                if '0' <= e <= '9':
                    agent_colors[ord(e) - ord('0')] = color
                elif 'A' <= e <= 'Z':
                    box_colors[ord(e) - ord('A')] = color
            line = server_messages.readline()
        
        # Read initial state.
        # line is currently "#initial".
        num_rows = 0
        num_cols = 0
        level_lines = []
        line = server_messages.readline()
        while not line.startswith('#'):
            level_lines.append(line)
            num_cols = max(num_cols, len(line))
            num_rows += 1
            line = server_messages.readline()

        num_agents = 0
        agent_rows = [None for _ in range(10)]
        agent_cols = [None for _ in range(10)]
        walls = [[False for _ in range(num_cols)] for _ in range(num_rows)]
        boxes = [['' for _ in range(num_cols)] for _ in range(num_rows)]
        row = 0
        for line in level_lines:
            for col, c in enumerate(line):
                if '0' <= c <= '9':
                    agent_rows[ord(c) - ord('0')] = row
                    agent_cols[ord(c) - ord('0')] = col
                    num_agents += 1
                elif 'A' <= c <= 'Z':
                    boxes[row][col] = c
                elif c == '+':
                    walls[row][col] = True
            
            row += 1
        del agent_rows[num_agents:]
        del agent_cols[num_agents:]
        
        # Read goal state.
        # line is currently "#goal".
        goals = [['' for _ in range(num_cols)] for _ in range(num_rows)]
        line = server_messages.readline()
        row = 0
        while not line.startswith('#'):
            for col, c in enumerate(line):
                if '0' <= c <= '9' or 'A' <= c <= 'Z':
                    goals[row][col] = c
            
            row += 1
            line = server_messages.readline()
        
        # End.
        # line is currently "#end".
            


        # import os
        # print(os.getcwd())
        # import time
        # import pyautogui
        # import os
        # import time
        # import pyautogui
        # import pygetwindow as gw
        # print('aaaaa',flush=True)
        # java_windows = [w for w in gw.getWindowsWithTitle('MAvis') if w.visible]
        # os.makedirs("screenshots", exist_ok=True)
        # if java_windows:
        #     print('bbb',flush=True)
        #     window = java_windows[0]
        #     bbox = (window.left+11, window.top+45, window.width-30, window.height-55)
        #     screenshot = pyautogui.screenshot(region=bbox)
        #     screenshot.save(f"screenshots/{levelname}.png")
        #     print(f"Saved screenshot for {levelname}",flush=True)
        # else:
        #     print(f"⚠️ Could not find the Java window for {levelname}",flush=True)




        
        State.agent_colors = agent_colors
        State.walls = walls
        State.box_colors = box_colors
        State.goals = goals

        # print(f"Part process level took {time.time() - start:.4f} seconds", flush=True)

        return State(agent_rows, agent_cols, boxes), levelname

    
    @staticmethod
    def print_search_status(start_time: 'int', explored: '{State, ...}', frontier: 'Frontier') -> None:
        status_template = '#Expanded: {:8,}, #Frontier: {:8,}, #Generated: {:8,}, Time: {:3.3f} s\n[Alloc: {:4.2f} MB, MaxAlloc: {:4.2f} MB]'
        elapsed_time = time.perf_counter() - start_time
        print(status_template.format(len(explored), frontier.size(), len(explored) + frontier.size(), elapsed_time, memory.get_usage(), memory.max_usage), file=sys.stderr, flush=True)

    @staticmethod
    def main(args) -> None:
        start_time = time.perf_counter()
        # Use stderr to print to the console.
        print('SearchClient initializing. I am sending this using the error output stream.', file=sys.stderr, flush=True)
        
        # Send client name to server.
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding='ASCII')

        print('SearchClient', flush=True)
        
        # We can also print comments to stdout by prefixing with a #.
        print('#This is a comment.', flush=True)

        # Parse the level.
        server_messages = sys.stdin
        if hasattr(server_messages, "reconfigure"):
            server_messages.reconfigure(encoding='ASCII')

        initial_state, levelname = SearchClient.parse_level(server_messages)

        # Select search strategy.
        frontier = None
        if args.bfs:
            frontier = FrontierBFS()
            method = 'bfs'
        elif args.dfs:
            frontier = FrontierDFS()
            method = 'dfs'
        elif args.astar:
            frontier = FrontierBestFirst(HeuristicAStar(initial_state))
            method = 'astar'
        elif args.wastar is not False:
            frontier = FrontierBestFirst(HeuristicWeightedAStar(initial_state, args.wastar))
            method = 'wastar'
        elif args.greedy:
            frontier = FrontierBestFirst(HeuristicGreedy(initial_state))
            method = 'greedy'
        elif args.iw:
            frontier = FrontierIW(HeuristicGreedy(initial_state))
            method = 'iw'
        # elif args.heuristicalternative:
            # frontier = FrontierAlternative(HeuristicAlternative(initial_state))
            # method = 'iw'
        # elif args.iwalternative:
            # frontier = FrontierIWAlternative(HeuristicAlternative(initial_state))
            # method = 'iw'
        elif args.siw:
            frontier = FrontierSIW(HeuristicGreedy(initial_state))
            method = 'siw'
        # elif args.bnovel:
            # frontier = FrontierBNovel(HeuristicAlternative(initial_state))
            # method = 'bnovel'
        elif args.bfws:
            frontier = Frontierbfws(HeuristicGreedy(initial_state))
            method = 'bfws'
        elif args.iwplus:
            frontier = FrontierIWplus(HeuristicGreedy(initial_state))
            method = 'iw'
        # elif args.na:
            # frontier = FrontierNA(HeuristicGreedy(initial_state))
            # method = 'na'
        elif args.oba:
            frontier = FrontierOBA(HeuristicOba(initial_state))
            method = 'oba'
        elif args.derelaxcolor:
            frontier = Frontierderelax(Heuristicderelax(initial_state))
            method = 'derelaxcolor'
        elif args.derelaxbox:
            frontier = Frontierderelax(Heuristicderelax(initial_state))
            method = 'derelaxbox'
        else:
            # Default to BFS search.
            frontier = FrontierBFS()
            method = 'bfs'
            print('Defaulting to BFS search. Use arguments -bfs, -dfs, -astar, -wastar, or -greedy to set the search strategy.', file=sys.stderr, flush=True)

        # Search for a plan.
        print('Starting {}.'.format(frontier.get_name()), file=sys.stderr, flush=True)
        plan, iw, state_space, time_plan = search(initial_state, frontier, method)




        # Print plan to server.
        if plan is None:
            print('Unable to solve level.', file=sys.stderr, flush=True)
            sys.exit(0)
        else:
            print('Found solution of length {}.'.format(len(plan)), file=sys.stderr, flush=True)
            
            for joint_action in plan:
                print("|".join(a.name_ + "@" + a.name_ for a in joint_action), flush=True)
                # We must read the server's response to not fill up the stdin buffer and block the server.
                response = server_messages.readline()


        # print('aaaa',time.perf_counter()-start_time, flush=True)

if __name__ == '__main__':

    # start = time.time()

    # Program arguments.
    parser = argparse.ArgumentParser(description='Simple client based on state-space graph search.')
    parser.add_argument('--max-memory', metavar='<MB>', type=float, default=2048.0, help='The maximum memory usage allowed in MB (soft limit, default 2048).')
    
    strategy_group = parser.add_mutually_exclusive_group()
    strategy_group.add_argument('-bfs', action='store_true', dest='bfs', help='Use the BFS strategy.')
    strategy_group.add_argument('-dfs', action='store_true', dest='dfs', help='Use the DFS strategy.')
    strategy_group.add_argument('-astar', action='store_true', dest='astar', help='Use the A* strategy.')
    strategy_group.add_argument('-wastar', action='store', dest='wastar', nargs='?', type=int, default=False, const=5, help='Use the WA* strategy.')
    strategy_group.add_argument('-greedy', action='store_true', dest='greedy', help='Use the Greedy strategy.')
    strategy_group.add_argument('-iw', action='store_true', dest='iw', help='Use the IW strategy.')
    strategy_group.add_argument('-heuristicalternative', action='store_true', dest='heuristicalternative', help='Use the heursitic alternative.')
    strategy_group.add_argument('-iwalternative', action='store_true', dest='iwalternative', help='Use the IW strategy with heursitic alternative.')
    strategy_group.add_argument('-siw', action='store_true', dest='siw', help='Use the SIW strategy.')
    strategy_group.add_argument('-bnovel', action='store_true', dest='bnovel', help='Use the BNovel strategy.')
    strategy_group.add_argument('-bfws', action='store_true', dest='bfws', help='Use the BFWS strategy.')
    strategy_group.add_argument('-iwplus', action='store_true', dest='iwplus', help='Use the IW+ strategy.')
    strategy_group.add_argument('-na', action='store_true', dest='na', help='Use the NA strategy.')
    strategy_group.add_argument('-oba', action='store_true', dest='oba', help='Use the OBA strategy.')
    strategy_group.add_argument('-derelaxcolor', action='store_true', dest='derelaxcolor', help='Use the DeRelaxation Color strategy.')
    strategy_group.add_argument('-derelaxbox', action='store_true', dest='derelaxbox', help='Use the DeRelaxation Box strategy.')




    args = parser.parse_args()
    
    # Set max memory usage allowed (soft limit).
    memory.max_usage = args.max_memory

    # Run client.
    SearchClient.main(args)
