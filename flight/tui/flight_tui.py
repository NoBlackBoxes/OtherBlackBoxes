import sys
import curses
import time
import tui_utils as tui

# Setup the curses screen window
screen = curses.initscr()
screen.nodelay(True)
curses.noecho()
curses.cbreak()

# Retrieve initial screen size
screen_y, screen_x = screen.getmaxyx()

# Define screen state
state = {'Rows': screen_y, 'Cols': screen_x}

# DEBUG: open commands file
commands_file = open("commands.txt")

# Control Loop
try:
    while True:
            
        # Get user input
        char = screen.getch()
        if char == ord('q'):
            sys.exit()
        elif char == ord('z'):
            break

        # DEBUG: read next command
        command = commands_file.readline()

        # Update
        tui.draw(screen, state)

        # Refresh
        screen.refresh()

        time.sleep(0.05)

finally:

    # DEBUG: close commands file
    commands_file.close()

    # Shutdown curses
    curses.nocbreak()
    screen.keypad(0)
    curses.echo()
    curses.endwin()

# FIN