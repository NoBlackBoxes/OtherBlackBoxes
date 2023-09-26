import sys
import curses
import time

def draw(screen, state):

    # Check if screen was resized
    resize = curses.is_term_resized(state['Rows'], state['Cols'])
    if resize:
        state['Rows'], state['Cols'] = screen.getmaxyx()
        screen.clear()
        curses.resizeterm(state['Rows'], state['Cols'])

    # Draw border
    screen.border()

    # Draw screen (terminal) size
    screen.addstr(0, 0, "rows: {0}, cols{1}".format(state['Rows'], state['Cols']))

    # Draw command prompt
    screen.addstr(state['Rows'] - 2, 1, "> ")
   
    return