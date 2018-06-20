"""Widgets for Curses-based CLI."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import curses
from tvm.tools.debug.ui import ui_common

RL = ui_common.RichLine
EXIT_TEXT = "exit"
HOME_TEXT = "HOME"
HELP_TEXT = "help"
TVM_DBG_BOX_BOTTOM = "-----------------"
NAVIGATION_MENU_COLOR_ATTR = "white_on_black"

class NavigationHistoryItem(object):
    """Individual item in navigation history."""

    def __init__(self, command, screen_output, scroll_position):
        """Constructor of NavigationHistoryItem.

        Args:
          command: (`str`) the command line text.
          screen_output: the screen output of the command.
          scroll_position: (`int`) scroll position in the screen output.
        """
        self.command = command
        self.screen_output = screen_output
        self.scroll_position = scroll_position


class CursesNavigationHistory(object):
    """Navigation history containing commands, outputs and scroll info."""

    BACK_ARROW_TEXT = "<--"
    FORWARD_ARROW_TEXT = "-->"

    def __init__(self, capacity):
        """Constructor of CursesNavigationHistory.

        Args:
          capacity: (`int`) How many items this object can hold. Each item consists
            of a command stirng, an output RichTextLines object and a scroll
            position.

        Raises:
          ValueError: If capacity is not a positive number.
        """
        if capacity <= 0:
            raise ValueError("In valid capacity value: %d" % capacity)

        self._capacity = capacity
        self._items = []
        self._pointer = -1

    def add_item(self, command, screen_output, scroll_position):
        """Add an item to the navigation histoyr.

        Args:
          command: command line text.
          screen_output: screen output produced for the command.
          scroll_position: (`int`) scroll position in the screen output.
        """
        if self._pointer + 1 < len(self._items):
            self._items = self._items[:self._pointer + 1]
        self._items.append(
            NavigationHistoryItem(command, screen_output, scroll_position))
        if len(self._items) > self._capacity:
            self._items = self._items[-self._capacity:]
        self._pointer = len(self._items) - 1

    def update_scroll_position(self, new_scroll_position):
        """Update the scroll position of the currently-pointed-to history item.

        Args:
          new_scroll_position: (`int`) new scroll-position value.

        Raises:
          ValueError: If the history is empty.
        """
        if not self._items:
            raise ValueError("Empty navigation history")
        self._items[self._pointer].scroll_position = new_scroll_position

    def size(self):
        """Get the size of widget items from list.

        Returns:
          ('int') length of widget items list.
        """
        return len(self._items)

    def pointer(self):
        """Get curses widget pointer.

        Returns:
          The pointer value.
        """
        return self._pointer

    def go_back(self):
        """Go back one place in the history, if possible.

        Decrease the pointer value by 1, if possible. Otherwise, the pointer value
        will be unchanged.

        Returns:
          The updated pointer value.

        Raises:
          ValueError: If history is empty.
        """
        if not self._items:
            raise ValueError("Empty navigation history")

        if self.can_go_back():
            self._pointer -= 1
        return self._items[self._pointer]

    def go_forward(self):
        """Go forward one place in the history, if possible.

        Increase the pointer value by 1, if possible. Otherwise, the pointer value
        will be unchanged.

        Returns:
          The updated pointer value.

        Raises:
          ValueError: If history is empty.
        """
        if not self._items:
            raise ValueError("Empty navigation history")

        if self.can_go_forward():
            self._pointer += 1
        return self._items[self._pointer]

    def can_go_back(self):
        """Test whether client can go back one place.

        Returns:
          (`bool`) Whether going back one place is possible.
        """
        return self._pointer >= 1

    def can_go_forward(self):
        """Test whether client can go forward one place.

        Returns:
          (`bool`) Whether going back one place is possible.
        """
        return self._pointer + 1 < len(self._items)

    def can_go_home(self):
        """Test whether client can go home place.

        Returns:
          (`bool`) Whether going back home place is possible.
        """
        if self._pointer >= 0:
            if self._items[self._pointer].command == "HOME":
                return False
            return True
        else:
            return False

    def can_go_help(self):
        """Test whether client can go help place.

        Returns:
          (`bool`) Whether going back help place is possible.
        """
        if self._pointer >= 0:
            if self._items[self._pointer].command == "help":
                return False
            return True
        else:
            return False

    def get_latest_command_info(self):
        """Get the latest command information.

        Returns:
          (`string`) Return the recent command shortcut.
        """
        return self._items[self._pointer].command

    def render(self,
               max_length,
               backward_command,
               forward_command,
               home_command,
               help_command,
               exit_command):
        """Render the rich text content of the single-line navigation bar.

        Args:
          max_length: (`int`) Maximum length of the navigation bar, in characters.
          backward_command: (`str`) command for going backward. Used to construct
            the shortcut menu item.
          forward_command: (`str`) command for going forward. Used to construct the
            shortcut menu item.

        Returns:
          (`ui_common.RichTextLines`) the navigation bar text with
            attributes.

        """
        output = RL("| ", NAVIGATION_MENU_COLOR_ATTR)
        output += RL(HOME_TEXT,
                     (ui_common.MenuItem(None, home_command,
                                         custom_color=NAVIGATION_MENU_COLOR_ATTR)
                      if self.can_go_home() else NAVIGATION_MENU_COLOR_ATTR))
        output += RL(" | ", NAVIGATION_MENU_COLOR_ATTR)

        output += RL(self.BACK_ARROW_TEXT,
                     (ui_common.MenuItem(None, backward_command,
                                         custom_color=NAVIGATION_MENU_COLOR_ATTR)
                      if self.can_go_back() else NAVIGATION_MENU_COLOR_ATTR))
        output += RL(" ", NAVIGATION_MENU_COLOR_ATTR)
        output += RL(self.FORWARD_ARROW_TEXT,
                     (ui_common.MenuItem(None, forward_command,
                                         custom_color=NAVIGATION_MENU_COLOR_ATTR)
                      if self.can_go_forward() else NAVIGATION_MENU_COLOR_ATTR))

        output_end = RL("| ", NAVIGATION_MENU_COLOR_ATTR)
        output_end += RL(HELP_TEXT,
                         (ui_common.MenuItem(None, help_command,
                                             custom_color=NAVIGATION_MENU_COLOR_ATTR)
                          if self.can_go_help() else NAVIGATION_MENU_COLOR_ATTR))
        output_end += RL(" | ", NAVIGATION_MENU_COLOR_ATTR)
        output_end += RL(EXIT_TEXT,
                         ui_common.MenuItem(None, exit_command,
                                            custom_color=NAVIGATION_MENU_COLOR_ATTR))
        output_end += RL(" |", NAVIGATION_MENU_COLOR_ATTR)

        output_middle = RL("", NAVIGATION_MENU_COLOR_ATTR)
        if (len(output)+len(output_end)) < max_length:
            space_need_size = max_length - len(output + output_end)
            if space_need_size > len(TVM_DBG_BOX_BOTTOM):
                space_need_size -= len(TVM_DBG_BOX_BOTTOM)
                space_middle_size = int(space_need_size/2)
                empty_line = (" " * (space_middle_size - (0 if space_need_size%2 else 1))
                              + TVM_DBG_BOX_BOTTOM + " " * space_middle_size)
                output_middle = RL(empty_line, NAVIGATION_MENU_COLOR_ATTR)
            else:
                empty_line = "-" * space_need_size
                output_middle = RL(empty_line, NAVIGATION_MENU_COLOR_ATTR)

        return ui_common.rich_text_lines_frm_line_list(
            [output + output_middle + output_end], additional_attr=curses.A_BOLD)
