# coding: utf-8
# pylint: disable=too-many-lines
"""Curses-Based Command-Line Interface of TVM Debugger (tvmdbg)."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import curses
from curses import textpad
import os
import signal
import sys
import threading

from tvm.tools.debug.ui import ui_base
from tvm.tools.debug.ui import ui_shared
from tvm.tools.debug.ui import command_parser
from tvm.tools.debug.ui import curses_widgets
from tvm.tools.debug.ui import ui_common
from tvm.tools.debug.ui import tensor_data


_SCROLL_REFRESH = "refresh"
_SCROLL_UP = "up"
_SCROLL_DOWN = "down"
_SCROLL_UP_A_LINE = "up_a_line"
_SCROLL_DOWN_A_LINE = "down_a_line"
_SCROLL_HOME = "home"
_SCROLL_END = "end"
_SCROLL_TO_LINE_INDEX = "scroll_to_line_index"

_COLOR_READY_COLORTERMS = ["gnome-terminal", "xfce4-terminal"]
_COLOR_ENABLED_TERM = "xterm-256color"
_TITLE_SIZE_DEFAULT = 1
_COMMAND_TEXT = "Command: "


def _get_cmd_from_line_attr_seg(mouse_x, attr_segs):
    """Attempt to extract command from the attribute segments of a line.

    Args:
      mouse_x: (int) x coordinate of the mouse event.
      attr_segs: (list) The list of attribute segments of a line from a
        RichTextLines object.

    Returns:
      (str or None) If a command exists: the command as a str; otherwise, None.
    """

    for seg in attr_segs:
        if seg[0] <= mouse_x < seg[1]:
            attributes = seg[2] if isinstance(seg[2], list) else [seg[2]]
            for attr in attributes:
                if isinstance(attr, ui_common.MenuItem):
                    return attr.content
    return None


class ScrollBar(object):
    """Vertical ScrollBar for Curses-based CLI.

    An object of this class has knowledge of the location of the scroll bar
    in the screen coordinates, the current scrolling position, and the total
    number of text lines in the screen text. By using this information, it
    can generate text rendering of the scroll bar, which consists of and UP
    button on the top and a DOWN button on the bottom, in addition to a scroll
    block in between, whose exact location is determined by the scrolling
    position. The object can also calculate the scrolling command (e.g.,
    _SCROLL_UP_A_LINE, _SCROLL_DOWN) from the coordinate of a mouse click
    event in the screen region it occupies.
    """

    BASE_ATTR = ui_shared.COLOR_WHITE + "_on_" + ui_shared.COLOR_BLACK
    BACKGROUND_BASE_ATTR = ui_shared.COLOR_WHITE + "_on_" + ui_shared.COLOR_BLACK

    def __init__(self,
                 min_x,
                 min_y,
                 max_x,
                 max_y,
                 scroll_position,
                 output_num_rows):
        """Constructor of ScrollBar.

        Args:
          min_x: (int) left index of the scroll bar on the screen (inclusive).
          min_y: (int) top index of the scroll bar on the screen (inclusive).
          max_x: (int) right index of the scroll bar on the screen (inclusive).
          max_y: (int) bottom index of the scroll bar on the screen (inclusive).
          scroll_position: (int) 0-based location of the screen output. For example,
            if the screen output is scrolled to the top, the value of
            scroll_position should be 0. If it is scrolled to the bottom, the value
            should be output_num_rows - 1.
          output_num_rows: (int) Total number of output rows.

        Raises:
          ValueError: If the width or height of the scroll bar, as determined
           by min_x, max_x, min_y and max_y, is too small.
        """

        self._min_x = min_x
        self._min_y = min_y
        self._max_x = max_x
        self._max_y = max_y
        self._scroll_position = scroll_position
        self._output_num_rows = output_num_rows
        self._scroll_bar_height = max_y - min_y + 1

        if self._max_x < self._min_x:
            raise ValueError("Insufficient width for ScrollBar (%d)" %
                             (self._max_x - self._min_x + 1))
        if self._max_y < self._min_y + 3:
            raise ValueError("Insufficient height for ScrollBar (%d)" %
                             (self._max_y - self._min_y + 1))

    def _block_y(self, screen_coord_sys=False):
        """Get the 0-based y coordinate of the scroll block.

        This y coordinate takes into account the presence of the UP and DN buttons
        present at the top and bottom of the ScrollBar. For example, at the home
        location, the return value will be 1; at the bottom location, the return
        value will be self._scroll_bar_height - 2.

        Args:
          screen_coord_sys: (`bool`) whether the return value will be in the
            screen coordinate system.

        Returns:
          (int) 0-based y coordinate of the scroll block, in the ScrollBar
            coordinate system by default. For example,
            when scroll position is at the top, this return value will be 1 (not 0,
            because of the presence of the UP button). When scroll position is at
            the bottom, this return value will be self._scroll_bar_height - 2
            (not self._scroll_bar_height - 1, because of the presence of the DOWN
            button).
        """

        rel_block_y = int(
            float(self._scroll_position) / (self._output_num_rows - 1) *
            (self._scroll_bar_height - 3)) + 1
        return rel_block_y + self._min_y if screen_coord_sys else rel_block_y

    def layout(self):
        """Get the RichTextLines layout of the scroll bar.

        Returns:
          (ui_common.RichTextLines) The text layout of the scroll bar.
        """
        width = self._max_x - self._min_x + 1
        empty_line = " " * width
        foreground_font_attr_segs = [(0, width, self.BASE_ATTR)]
        background_font_attr_segs = [(0, width, self.BACKGROUND_BASE_ATTR)]

        if self._output_num_rows > 1:
            block_y = self._block_y()

            if width == 1:
                up_text = "^"
                down_text = "v"
            elif width == 2:
                up_text = "^ "
                down_text = "v "
            elif width == 3:
                up_text = " ^ "
                down_text = " v "
            else:
                up_text = " ^ "
                down_text = " v "

            layout = ui_common.RichTextLines(
                [up_text], font_attr_segs={0: [(0, width, self.BASE_ATTR)]})
            for i in range(1, self._scroll_bar_height - 1):
                font_attr_segs = background_font_attr_segs
                if i == block_y:
                    font_attr_segs = None
                layout.append("  ", font_attr_segs=font_attr_segs)
            layout.append(down_text, font_attr_segs=foreground_font_attr_segs)
        else:
            layout = ui_common.RichTextLines(
                [empty_line] * self._scroll_bar_height)

        return layout

    def _get_click_command(self, mouse_y):
        if self._output_num_rows <= 1:
            return None
        elif mouse_y == self._min_y:
            return _SCROLL_UP_A_LINE
        elif mouse_y == self._max_y:
            return _SCROLL_DOWN_A_LINE
        elif (mouse_y > self._block_y(screen_coord_sys=True) and
              mouse_y < self._max_y):
            return _SCROLL_DOWN
        elif (mouse_y < self._block_y(screen_coord_sys=True) and
              mouse_y > self._min_y):
            return _SCROLL_UP
        return None


def _screen_getmouse():
    return curses.getmouse()


def _screen_new_output_pad(rows, cols):
    """Generate a new pad on the screen.

    Args:
      rows: (int) Number of rows the pad will have: not limited to screen size.
      cols: (int) Number of columns the pad will have: not limited to screen
        size.

    Returns:
      A curses textpad object.
    """

    return curses.newpad(rows, cols)


def _refresh_pad(pad, *args):
    try:
        pad.refresh(*args)
    except curses.error:
        pass


def _format_indices(indices):
    # Remove the spaces to make it compact.
    return repr(indices).replace(" ", "")


class CursesUI(ui_base.BaseUI):
    """Curses-based Command-line UI.

    In this class, the methods with the prefix "_screen_" are the methods that
    interact with the actual terminal using the curses library.
    """

    CLI_TERMINATOR_KEY = 7  # Terminator key for input text box.
    CLI_TAB_KEY = ord("\t")
    BACKSPACE_KEY = ord("\b")
    REGEX_SEARCH_PREFIX = "/"
    TENSOR_INDICES_NAV_PREFIX = "@"

    _NAVIGATION_FORWARD_COMMAND = "next"
    _NAVIGATION_BACK_COMMAND = "prev"
    _NAVIGATION_EXIT_COMMAND = "exit"
    _NAVIGATION_HELP_COMMAND = "help"
    _NAVIGATION_HOME_COMMAND = "HOME"

    # Limit screen width to work around the limitation of the curses library that
    # it may return invalid x coordinates for large values.
    _SCREEN_WIDTH_LIMIT = 220

    # Possible Enter keys. 343 is curses key code for the num-pad Enter key when
    # num lock is off.
    CLI_CR_KEYS = [ord("\n"), ord("\r"), 343]

    _KEY_MAP = {
        127: curses.KEY_BACKSPACE,  # Backspace
        curses.KEY_DC: 4,  # Delete
    }

    _FOREGROUND_COLORS = {
        ui_shared.COLOR_WHITE: curses.COLOR_WHITE,
        ui_shared.COLOR_RED: curses.COLOR_RED,
        ui_shared.COLOR_GREEN: curses.COLOR_GREEN,
        ui_shared.COLOR_YELLOW: curses.COLOR_YELLOW,
        ui_shared.COLOR_BLUE: curses.COLOR_BLUE,
        ui_shared.COLOR_CYAN: curses.COLOR_CYAN,
        ui_shared.COLOR_MAGENTA: curses.COLOR_MAGENTA,
        ui_shared.COLOR_BLACK: curses.COLOR_BLACK,
    }
    _BACKGROUND_COLORS = {
        "transparent": -1,
        ui_shared.COLOR_WHITE: curses.COLOR_WHITE,
        ui_shared.COLOR_BLACK: curses.COLOR_BLACK,
    }

    # Font attribute for search and highlighting.
    _SEARCH_HIGHLIGHT_FONT_ATTR = (
        ui_shared.COLOR_BLACK + "_on_" + ui_shared.COLOR_WHITE)
    _ARRAY_INDICES_COLOR_PAIR = (
        ui_shared.COLOR_BLACK + "_on_" + ui_shared.COLOR_WHITE)
    _ERROR_TOAST_COLOR_PAIR = (
        ui_shared.COLOR_RED + "_on_" + ui_shared.COLOR_WHITE)
    _INFO_TOAST_COLOR_PAIR = (
        ui_shared.COLOR_BLUE + "_on_" + ui_shared.COLOR_WHITE)
    _STATUS_BAR_COLOR_PAIR = (
        ui_shared.COLOR_WHITE + "_on_" + ui_shared.COLOR_BLACK)
    _UI_WAIT_COLOR_PAIR = (
        ui_shared.COLOR_MAGENTA + "_on_" + ui_shared.COLOR_WHITE)
    _NAVIGATION_WARNING_COLOR_PAIR = (
        ui_shared.COLOR_RED + "_on_" + ui_shared.COLOR_WHITE)

    _UI_WAIT_MESSAGE = "Processing..."

    def __init__(self, on_ui_exit=None, config=None):
        """Constructor of CursesUI.

        Args:
          on_ui_exit: (Callable) Callback invoked when the UI exits.
          config: An instance of `ui_config.CLIConfig()` carrying user-facing
            configurations.
        """

        ui_base.BaseUI.__init__(self, on_ui_exit=on_ui_exit, config=config)

        self._candidates_top_row = None
        self._cmd_ptr = None
        self._color_pairs = None
        self._command_textbox = None
        self._command_window = None
        self._curr_search_regex = None
        self._default_color_pair = None
        self._main_menu = None
        self._main_menu_pad = None
        self._mouse_enabled = None
        self._nav_bar_pad = None
        self._nav_bar = None
        self._output_pad_height = None
        self._output_pad_width = None
        self._scroll_bar = None
        self._scroll_info = None
        self._textbox_curr_terminator = None
        self._textbox_pending_cmd_changed = None
        self._unwrapped_regex_match_lines = None
        self._title_color = "white_on_black"
        self._single_instance_lock = threading.Lock()

        self._screen_init()
        self._screen_refresh_size()
        # Initialize some UI component size and locations.
        self._init_layout()

        self._command_history_store = ui_common.CommandHistory()

        # Active list of command history, used in history navigation.
        # _command_handler_registry holds all the history commands the CLI has
        # received, up to a size limit. _active_command_history is the history
        # currently being navigated in, e.g., using the Up/Down keys. The latter
        # can be different from the former during prefixed or regex-based history
        # navigation, e.g., when user enter the beginning of a command and hit Up.
        self._active_command_history = []

        # Pointer to the current position in the history sequence.
        # 0 means it is a new command being keyed in.
        self._command_pointer = 0

        self._command_history_limit = 100

        self._pending_command = ""

        self._nav_history = curses_widgets.CursesNavigationHistory(10)

        # State related to screen output.
        self._output_pad = None
        self._output_pad_row = 0
        self._output_array_pointer_indices = None
        self._curr_unwrapped_output = None
        self._curr_wrapped_output = None

        try:
            # Register signal handler for SIGINT.
            signal.signal(signal.SIGINT, self._interrupt_handler)
        except ValueError:
            # Running in a child thread, can't catch signals.
            pass

    def _init_layout(self):
        """Initialize the layout of UI components.

        Initialize the location and size of UI components such as command textbox
        and output region according to the terminal size.
        """

        # NamedTuple for rectangular locations on screen
        self.rectangle = collections.namedtuple("rectangle",
                                                "top left bottom right")

        # Height of command text box
        self._command_textbox_height = 2

        self._title_row = 0

        # Row index of the Navigation Bar (i.e., the bar that contains forward and
        # backward buttons and displays the current command line).
        self._nav_bar_row = 1 + _TITLE_SIZE_DEFAULT

        # Top row index of the output pad.
        # A "pad" is a curses object that holds lines of text and not limited to
        # screen size. It can be rendered on the screen partially with scroll
        # parameters specified.
        self._output_top_row = 2 + _TITLE_SIZE_DEFAULT

        # Number of rows that the output pad has.
        self._output_num_rows = (
            self._max_y - self._output_top_row - self._command_textbox_height - 1)

        # Row index of scroll information line: Taking into account the zero-based
        # row indexing and the command textbox area under the scroll information
        # row.
        self._output_scroll_row = self._max_y - 1 - self._command_textbox_height

        # Tab completion bottom row.
        self._candidates_top_row = self._output_scroll_row - 4
        self._candidates_bottom_row = self._output_scroll_row - 1

        # Maximum number of lines the candidates display can have.
        # Daya: Review
        self._candidates_max_lines = int(self._output_num_rows / 2) + 2

        self.max_output_lines = 10000

        # Regex search state.
        self._curr_search_regex = None
        self._unwrapped_regex_match_lines = []

        # Size of view port on screen, which is always smaller or equal to the
        # screen size.
        self._output_pad_screen_height = self._output_num_rows - 1
        self._output_pad_screen_width = self._max_x - 2
        self._output_pad_screen_location = self.rectangle(
            top=self._output_top_row,
            left=0,
            bottom=self._output_top_row + self._output_num_rows,
            right=self._output_pad_screen_width)

    def _screen_init(self):
        """Screen initialization.

        Creates curses stdscr and initialize the color pairs for display.
        """
        # If the terminal type is color-ready, enable it.
        if os.getenv("COLORTERM") in _COLOR_READY_COLORTERMS:
            os.environ["TERM"] = _COLOR_ENABLED_TERM
        self._stdscr = curses.initscr()
        self._command_window = None
        self._screen_color_init()

    def _screen_color_init(self):
        """Initialization of screen colors."""
        curses.start_color()
        curses.use_default_colors()
        self._color_pairs = {}
        color_index = 0

        # Prepare color pairs.
        for fg_color in self._FOREGROUND_COLORS:
            for bg_color in self._BACKGROUND_COLORS:
                color_index += 1

                if bg_color == ui_shared.COLOR_GRAY:
                    curses.init_pair(color_index, self._FOREGROUND_COLORS[fg_color], 236)
                else:
                    curses.init_pair(color_index, self._FOREGROUND_COLORS[fg_color],
                                     self._BACKGROUND_COLORS[bg_color])

                color_name = fg_color
                if bg_color != "transparent":
                    color_name += "_on_" + bg_color

                self._color_pairs[color_name] = curses.color_pair(color_index)

        # Try getting color(s) available only under 256-color support.
        try:
            color_index += 1
            curses.init_pair(color_index, 245, -1)
            self._color_pairs[ui_shared.COLOR_GRAY] = curses.color_pair(color_index)
        except curses.error:
            # Use fall-back color(s):
            self._color_pairs[ui_shared.COLOR_GRAY] = (
                self._color_pairs[ui_shared.COLOR_GREEN])

        # A_BOLD or A_BLINK is not really a "color". But place it here for
        # convenience.
        self._color_pairs["bold"] = curses.A_BOLD
        self._color_pairs["blink"] = curses.A_BLINK
        self._color_pairs["underline"] = curses.A_UNDERLINE

        # Default color pair to use when a specified color pair does not exist.
        self._default_color_pair = self._color_pairs[ui_shared.COLOR_BLUE]

    def _screen_launch(self):
        """Launch the curses screen."""

        curses.noecho()
        curses.cbreak()
        self._stdscr.keypad(1)

        self._mouse_enabled = self.config.get("mouse_mode")
        self._screen_set_mousemask()
        self.config.set_callback(
            "mouse_mode",
            lambda cfg: self._set_mouse_enabled(cfg.get("mouse_mode")))

        self._screen_create_command_window()

    def _screen_create_command_window(self):
        """Create command window according to screen size."""
        if self._command_window:
            del self._command_window

        self._command_window = curses.newwin(
            self._command_textbox_height, self._max_x - len(self.CLI_PROMPT),
            self._max_y - self._command_textbox_height,
            len(self.CLI_PROMPT))

    def _screen_refresh(self):
        self._stdscr.refresh()

    def _screen_terminate(self):
        """Terminate the curses screen."""

        self._stdscr.keypad(0)
        curses.nocbreak()
        curses.echo()
        curses.endwin()

        try:
            # Remove SIGINT handler.
            signal.signal(signal.SIGINT, signal.SIG_DFL)
        except ValueError:
            # Can't catch signals unless you're the main thread.
            pass

    def run_ui(self,
               init_command=None,
               title=None,
               title_color=None):
        """Run the CLI: See the doc of ui_base.BaseUI.run_ui for more details."""

        # Only one instance of the Curses UI can be running at a time, since
        # otherwise they would try to both read from the same keystrokes, and write
        # to the same screen.
        self._single_instance_lock.acquire()

        self._screen_launch()

        # Optional initial command.
        if init_command is not None:
            self._dispatch_command(init_command)

        if title is not None:
            self._title()

        # CLI main loop.
        exit_token = self._ui_loop()

        if self._on_ui_exit:
            self._on_ui_exit()

        self._screen_terminate()

        self._single_instance_lock.release()

        return exit_token

    def get_help(self):
        """ Retrieve ArgumentParser full help text.
        """
        return self._command_handler_registry.get_help()

    def _addstr(self, *args):
        try:
            self._stdscr.addstr(*args)
        except curses.error:
            pass

    def _screen_create_command_textbox(self, existing_command=None):
        """Create command textbox on screen.

        Args:
          existing_command: (str) A command string to put in the textbox right
            after its creation.
        """

        # Display the tvmdbg prompt.
        self._addstr(self._max_y - self._command_textbox_height, 0,
                     self.CLI_PROMPT, curses.A_BOLD)
        self._stdscr.refresh()

        self._command_window.clear()

        # Command text box.
        self._command_textbox = textpad.Textbox(
            self._command_window, insert_mode=True)

        # Enter existing command.
        self._auto_key_in(existing_command)

    def _ui_loop(self):
        """Command-line UI loop.

        Returns:
          An exit token of arbitrary type. The token can be None.
        """

        while True:
            # Enter history command if pointer is in history (> 0):
            if self._command_pointer > 0:
                existing_command = self._active_command_history[-self._command_pointer]
            else:
                existing_command = self._pending_command
            self._screen_create_command_textbox(existing_command)

            try:
                command, terminator, pending_command_changed = self._get_user_command()
            except ui_common.CommandLineExit as exception:
                return exception.exit_token

            if not command and terminator != self.CLI_TAB_KEY:
                continue

            if terminator in self.CLI_CR_KEYS or terminator == curses.KEY_MOUSE:
                exit_token = self._dispatch_command(command)
                if exit_token is not None:
                    return exit_token
            elif terminator == self.CLI_TAB_KEY:
                tab_completed = self._tab_complete(command)
                self._pending_command = tab_completed
                self._cmd_ptr = 0
            elif pending_command_changed:
                self._pending_command = command

        return None

    def _get_user_command(self):
        """Get user command from UI.

        Returns:
          command: (str) The user-entered command.
          terminator: (str) Terminator type for the command.
            If command is a normal command entered with the Enter key, the value
            will be the key itself. If this is a tab completion call (using the
            Tab key), the value will reflect that as well.
          pending_command_changed:  (bool) If the pending command has changed.
            Used during command history navigation.
        """

        # First, reset textbox state variables.
        self._textbox_curr_terminator = None
        self._textbox_pending_cmd_changed = False

        command = self._screen_get_user_command()
        command = self._strip_terminator(command)
        return (command, self._textbox_curr_terminator,
                self._textbox_pending_cmd_changed)

    def _screen_get_user_command(self):
        return self._command_textbox.edit(validate=self._on_textbox_keypress)

    def _strip_terminator(self, command):
        if not command:
            return command

        for key in self.CLI_CR_KEYS:
            if key < 256:
                command = command.replace(chr(key), "")

        return command.strip()

    def _screen_refresh_size(self):
        self._max_y, self._max_x = self._stdscr.getmaxyx()
        if self._max_x > self._SCREEN_WIDTH_LIMIT:
            self._max_x = self._SCREEN_WIDTH_LIMIT

    def _navigate_screen_output(self, command):
        """Navigate in screen output history.

        Args:
          command: (`str`) the navigation command, from
            {self._NAVIGATION_FORWARD_COMMAND, self._NAVIGATION_BACK_COMMAND}.
        """
        if command == self._NAVIGATION_FORWARD_COMMAND:
            if self._nav_history.can_go_forward():
                item = self._nav_history.go_forward()
                scroll_position = item.scroll_position
            else:
                self._toast("At the LATEST in navigation history!",
                            color=self._NAVIGATION_WARNING_COLOR_PAIR)
                return
        else:
            if self._nav_history.can_go_back():
                item = self._nav_history.go_back()
                scroll_position = item.scroll_position
            else:
                self._toast("At the OLDEST in navigation history!",
                            color=self._NAVIGATION_WARNING_COLOR_PAIR)
                return

        self._display_output(item.screen_output)
        if scroll_position != 0:
            self._scroll_output(_SCROLL_TO_LINE_INDEX, line_index=scroll_position)

    def _dispatch_command(self, command):
        """Dispatch user command.

        Args:
          command: (str) Command to dispatch.

        Returns:
          An exit token object. None value means that the UI loop should not exit.
          A non-None value means the UI loop should exit.
        """

        if self._output_pad:
            self._toast(self._UI_WAIT_MESSAGE, color=self._UI_WAIT_COLOR_PAIR)

        if command in self.CLI_EXIT_COMMANDS:
            # Explicit user command-triggered exit: EXPLICIT_USER_EXIT as the exit
            # token.
            return ui_common.EXPLICIT_USER_EXIT
        elif (command == self._NAVIGATION_FORWARD_COMMAND or
              command == self._NAVIGATION_BACK_COMMAND):
            self._navigate_screen_output(command)
            return None

        if command:
            self._command_history_store.add_command(command)

        if (command.startswith(self.REGEX_SEARCH_PREFIX) and
                self._curr_unwrapped_output):
            if len(command) > len(self.REGEX_SEARCH_PREFIX):
                # Command is like "/regex". Perform regex search.
                regex = command[len(self.REGEX_SEARCH_PREFIX):]

                self._curr_search_regex = regex
                self._display_output(self._curr_unwrapped_output, highlight_regex=regex)
            elif self._unwrapped_regex_match_lines:
                # Command is "/". Continue scrolling down matching lines.
                self._display_output(
                    self._curr_unwrapped_output,
                    is_refresh=True,
                    highlight_regex=self._curr_search_regex)

            self._command_pointer = 0
            self._pending_command = ""
            return None
        elif command.startswith(self.TENSOR_INDICES_NAV_PREFIX):
            indices_str = command[1:].strip()
            if indices_str:
                try:
                    indices = command_parser.parse_indices(indices_str)
                    omitted, line_index, _, _ = tensor_data.locate_tensor_element(
                        self._curr_wrapped_output, indices)
                    if not omitted:
                        self._scroll_output(
                            _SCROLL_TO_LINE_INDEX, line_index=line_index)
                except Exception as exception:  # pylint: disable=broad-except
                    self._error_toast(str(exception))
            else:
                self._error_toast("Empty indices.")

            return None

        try:
            prefix, args, output_file_path = ui_base._parse_command(command)
        except SyntaxError as exception:
            self._error_toast(str(exception))
            return None

        if not prefix:
            # Empty command: take no action. Should not exit.
            return None

        # Take into account scroll bar width.
        screen_info = {"cols": self._max_x - 2}
        exit_token = None
        if self._command_handler_registry.is_registered(prefix):
            try:
                screen_output = self._command_handler_registry.dispatch_command(
                    prefix, args, screen_info=screen_info)
            except ui_common.CommandLineExit as exception:
                exit_token = exception.exit_token
        else:
            screen_output = ui_common.RichTextLines([
                self.ERROR_MESSAGE_PREFIX + "Invalid command prefix \"%s\"" % prefix
            ])

        # Clear active command history. Until next up/down history navigation
        # occurs, it will stay empty.
        self._active_command_history = []

        if exit_token is not None:
            return exit_token

        self._nav_history.add_item(command, screen_output, 0)

        self._display_output(screen_output)
        if output_file_path:
            try:
                screen_output.write_to_file(output_file_path)
                self._info_toast("Wrote output to %s" % output_file_path)
            except Exception:  # pylint: disable=broad-except
                self._error_toast("Failed to write output to %s" % output_file_path)

        self._command_pointer = 0
        self._pending_command = ""
        return None

    def _screen_gather_textbox_str(self):
        """Gather the text string in the command text box.

        Returns:
          (str) the current text string in the command textbox, excluding any
          return keys.
        """

        txt = self._command_textbox.gather()
        return txt.strip()

    def _on_textbox_keypress(self, key_code):
        """Text box key validator: Callback of key strokes.

        Handles a user's keypress in the input text box. Translates certain keys to
        terminator keys for the textbox to allow its edit() method to return.
        Also handles special key-triggered events such as PgUp/PgDown scrolling of
        the screen output.

        Args:
          key_code: (int) Key code.

        Returns:
          (int) A translated key code. In most cases, this is identical to the
            input x. However, if x is a Return key, the return value will be
            CLI_TERMINATOR_KEY, so that the text box's edit() method can return.

        Raises:
          TypeError: If the input x is not of type int.
          ui_common.CommandLineExit: If a mouse-triggered command returns
            an exit token when dispatched.
        """
        if not isinstance(key_code, int):
            raise TypeError("Key validator expected type int, received type %s" %
                            type(key_code))

        if key_code in self.CLI_CR_KEYS:
            # Make Enter key the terminator
            self._textbox_curr_terminator = key_code
            return self.CLI_TERMINATOR_KEY
        elif key_code == self.CLI_TAB_KEY:
            self._textbox_curr_terminator = self.CLI_TAB_KEY
            return self.CLI_TERMINATOR_KEY
        elif key_code == curses.KEY_PPAGE:
            self._scroll_output(_SCROLL_UP_A_LINE)
            return key_code
        elif key_code == curses.KEY_NPAGE:
            self._scroll_output(_SCROLL_DOWN_A_LINE)
            return key_code
        elif key_code == curses.KEY_HOME:
            self._scroll_output(_SCROLL_HOME)
            return key_code
        elif key_code == curses.KEY_END:
            self._scroll_output(_SCROLL_END)
            return key_code
        elif key_code in [curses.KEY_UP, curses.KEY_DOWN]:
            # Command history navigation.
            if not self._active_command_history:
                hist_prefix = self._screen_gather_textbox_str()
                self._active_command_history = (
                    self._command_history_store.lookup_prefix(
                        hist_prefix, self._command_history_limit))

            if self._active_command_history:
                if key_code == curses.KEY_UP:
                    if self._command_pointer < len(self._active_command_history):
                        self._command_pointer += 1
                elif key_code == curses.KEY_DOWN:
                    if self._command_pointer > 0:
                        self._command_pointer -= 1
            else:
                self._command_pointer = 0

            self._textbox_curr_terminator = key_code

            # Force return from the textbox edit(), so that the textbox can be
            # redrawn with a history command entered.
            return self.CLI_TERMINATOR_KEY
        elif key_code == curses.KEY_RESIZE:
            # Respond to terminal resize.
            self._screen_refresh_size()
            self._init_layout()
            self._screen_create_command_window()
            self._redraw_output()

            # Force return from the textbox edit(), so that the textbox can be
            # redrawn.
            return self.CLI_TERMINATOR_KEY
        elif key_code == curses.KEY_MOUSE and self._mouse_enabled:
            try:
                _, mouse_x, mouse_y, _, mouse_event_type = _screen_getmouse()
            except curses.error:
                mouse_event_type = None

            if mouse_event_type == curses.BUTTON1_RELEASED:
                # Logic for mouse-triggered scrolling.
                if mouse_x >= self._max_x - 2:
                    scroll_command = self._scroll_bar._get_click_command(mouse_y)
                    if scroll_command is not None:
                        self._scroll_output(scroll_command)
                    return key_code
                else:
                    command = self._output_hyperlink_command(mouse_x, mouse_y)
                    if command:
                        self._screen_create_command_textbox()
                        exit_token = self._dispatch_command(command)
                        if exit_token is not None:
                            raise ui_common.CommandLineExit(exit_token=exit_token)
        # Mark the pending command as modified.
        self._textbox_pending_cmd_changed = True
        # Invalidate active command history.
        self._command_pointer = 0
        self._active_command_history = []
        return self._KEY_MAP.get(key_code, key_code)

    def _redraw_output(self):
        if self._curr_unwrapped_output is not None:
            self._display_nav_bar()
            self._display_main_menu(self._curr_unwrapped_output)
            self._display_output(self._curr_unwrapped_output, is_refresh=True)

    def _output_hyperlink_command(self, mouse_x, mouse_y):
        output_top = self._output_top_row
        if self._main_menu_pad:
            output_top += 1

        if mouse_y == self._nav_bar_row and self._nav_bar:
            # Click was in the nav bar.
            return _get_cmd_from_line_attr_seg(mouse_x,
                                               self._nav_bar.font_attr_segs[0])
        elif mouse_y == self._output_top_row and self._main_menu_pad:
            # Click was in the menu bar.
            return _get_cmd_from_line_attr_seg(mouse_x,
                                               self._main_menu.font_attr_segs[0])
        absolute_mouse_y = mouse_y + self._output_pad_row - output_top
        if absolute_mouse_y in self._curr_wrapped_output.font_attr_segs:
            return _get_cmd_from_line_attr_seg(
                mouse_x, self._curr_wrapped_output.font_attr_segs[absolute_mouse_y])
        return None

    def _title(self):
        """Display title.

        Args:
          title: (str) The title to display.
          title_color: (str) Color of the title, e.g., "yellow".
        """

        tvm_dgb_box_top = "-----------------"
        tvm_dgb_txt = "    TVM DEBUG    "
        space_need_half = int(((self._max_x) - len(tvm_dgb_txt))/2)
        tvm_dgb_box_top = (" " * space_need_half) + tvm_dgb_box_top + (" " * (space_need_half))
        tvm_dgb_txt = (" " * space_need_half) + tvm_dgb_txt + (" " * space_need_half)
        #tvm_dgb_box_bottom = (" " * txt_begin) + tvm_dgb_box_bottom + (" " * txt_begin)
        if len(tvm_dgb_txt) < (self._max_x):
            tvm_dgb_box_top += " "
            tvm_dgb_txt += " "
        #    #tvm_dgb_box_bottom += " "
        tvm_dgb_box_top = tvm_dgb_box_top[:len(tvm_dgb_box_top)-1]
        tvm_dgb_txt = tvm_dgb_txt[:len(tvm_dgb_txt)-1]

        self._screen_draw_text_line(
            self._title_row, tvm_dgb_box_top, attr=curses.A_NORMAL | curses.A_BOLD,
            color=self._title_color)
        self._screen_draw_text_line(
            self._title_row + 1, tvm_dgb_txt, attr=curses.A_NORMAL | curses.A_BOLD,
            color=self._title_color)
        #self._screen_draw_text_line(
        #    self._title_row + 2, tvm_dgb_box_bottom, attr=curses.A_NORMAL | curses.A_BOLD,
        #    color=self._title_color)

    def _auto_key_in(self, command, erase_existing=False):
        """Automatically key in a command to the command Textbox.

        Args:
          command: The command, as a string or None.
          erase_existing: (bool) whether existing text (if any) is to be erased
              first.
        """
        if erase_existing:
            self._erase_existing_command()

        command = command or ""
        for cmd in command:
            self._command_textbox.do_command(ord(cmd))

    def _erase_existing_command(self):
        """Erase existing text in command textpad."""

        existing_len = len(self._command_textbox.gather())
        for _ in range(existing_len):
            self._command_textbox.do_command(self.BACKSPACE_KEY)

    def _screen_draw_text_line(self, row, line, attr=curses.A_NORMAL, color=None):
        """Render a line of text on the screen.

        Args:
          row: (int) Row index.
          line: (str) The line content.
          attr: curses font attribute.
          color: (str) font foreground color name.

        Raises:
          TypeError: If row is not of type int.
        """

        if not isinstance(row, int):
            raise TypeError("Invalid type in row")

        if len(line) > self._max_x:
            line = line[:self._max_x]

        color_pair = (self._default_color_pair if color is None else
                      self._color_pairs[color])

        self._addstr(row, 0, line, color_pair | attr)
        self._screen_refresh()

    def _screen_display_output(self, output):
        """Actually render text output on the screen.

        Wraps the lines according to screen width. Pad lines below according to
        screen height so that the user can scroll the output to a state where
        the last non-empty line is on the top of the screen. Then renders the
        lines on the screen.

        Args:
          output: (RichTextLines) text lines to display on the screen. These lines
            may have widths exceeding the screen width. This method will take care
            of the wrapping.

        Returns:
          (List of int) A list of line indices, in the wrapped output, where there
            are regex matches.
        """

        # Wrap the output lines according to screen width.
        self._curr_wrapped_output, wrapped_line_indices = (
            ui_common.wrap_rich_text_lines(output, self._max_x - 2))

        # Append lines to curr_wrapped_output so that the user can scroll to a
        # state where the last text line is on the top of the output area.
        self._curr_wrapped_output.lines.extend([""] * (self._output_num_rows - 1))

        # Limit number of lines displayed to avoid curses overflow problems.
        if self._curr_wrapped_output.num_lines() > self.max_output_lines:
            self._curr_wrapped_output = self._curr_wrapped_output.slice(
                0, self.max_output_lines)
            self._curr_wrapped_output.lines.append("Output cut off at %d lines!" %
                                                   self.max_output_lines)
            self._curr_wrapped_output.font_attr_segs[self.max_output_lines] = [
                (0, len(output.lines[-1]), ui_shared.COLOR_MAGENTA)
            ]

        self._display_nav_bar()
        self._display_main_menu(self._curr_wrapped_output)

        (self._output_pad, self._output_pad_height,
         self._output_pad_width) = self._display_lines(self._curr_wrapped_output,
                                                       self._output_num_rows)

        # The indices of lines with regex matches (if any) need to be mapped to
        # indices of wrapped lines.
        return [
            wrapped_line_indices[line]
            for line in self._unwrapped_regex_match_lines
        ]

    def _display_output(self, output, is_refresh=False, highlight_regex=None):
        """Display text output in a scrollable text pad.

        This method does some preprocessing on the text lines, render them on the
        screen and scroll to the appropriate line. These are done according to regex
        highlighting requests (if any), scroll-to-next-match requests (if any),
        and screen refresh requests (if any).

        Args:
          output: A RichTextLines object that is the screen output text.
          is_refresh: (bool) Is this a refreshing display with existing output.
          highlight_regex: (str) Optional string representing the regex used to
            search and highlight in the current screen output.
        """

        if not output:
            return

        if highlight_regex:
            try:
                output = ui_common.regex_find(
                    output, highlight_regex, font_attr=self._SEARCH_HIGHLIGHT_FONT_ATTR)
            except ValueError as exception:
                self._error_toast(str(exception))
                return

            if not is_refresh:
                # Perform new regex search on the current output.
                self._unwrapped_regex_match_lines = output.annotations[
                    ui_common.REGEX_MATCH_LINES_KEY]
            else:
                # Continue scrolling down.
                self._output_pad_row += 1
        else:
            self._curr_unwrapped_output = output
            self._unwrapped_regex_match_lines = []

        # Display output on the screen.
        wrapped_regex_match_lines = self._screen_display_output(output)

        # Now that the text lines are displayed on the screen scroll to the
        # appropriate line according to previous scrolling state and regex search
        # and highlighting state.

        if highlight_regex:
            next_match_line = -1
            for match_line in wrapped_regex_match_lines:
                if match_line >= self._output_pad_row:
                    next_match_line = match_line
                    break

            if next_match_line >= 0:
                self._scroll_output(
                    _SCROLL_TO_LINE_INDEX, line_index=next_match_line)
            else:
                # Regex search found no match >= current line number. Display message
                # stating as such.
                self._toast("Pattern not found", color=self._ERROR_TOAST_COLOR_PAIR)
        elif is_refresh:
            self._scroll_output(_SCROLL_REFRESH)
        elif ui_common.INIT_SCROLL_POS_KEY in output.annotations:
            line_index = output.annotations[ui_common.INIT_SCROLL_POS_KEY]
            self._scroll_output(_SCROLL_TO_LINE_INDEX, line_index=line_index)
        else:
            self._output_pad_row = 0
            self._scroll_output(_SCROLL_HOME)

    def _display_lines(self, output, min_num_rows):
        """Display RichTextLines object on screen.

        Args:
          output: A RichTextLines object.
          min_num_rows: (int) Minimum number of output rows.

        Returns:
          1) The text pad object used to display the main text body.
          2) (int) number of rows of the text pad, which may exceed screen size.
          3) (int) number of columns of the text pad.

        Raises:
          ValueError: If input argument "output" is invalid.
        """

        if not isinstance(output, ui_common.RichTextLines):
            raise ValueError(
                "Output is required to be an instance of RichTextLines, but is not.")

        self._screen_refresh()

        # Number of rows the output area will have.
        rows = max(min_num_rows, len(output.lines))

        # Size of the output pad, which may exceed screen size and require
        # scrolling.
        cols = self._max_x - 2

        # Create new output pad.
        pad = _screen_new_output_pad(rows, cols)

        for i in range(len(output.lines)):
            if i in output.font_attr_segs:
                self._screen_add_line_to_output_pad(
                    pad, i, output.lines[i], color_segments=output.font_attr_segs[i],
                    additional_attr=output.font_additional_attr)
            else:
                self._screen_add_line_to_output_pad(pad, i, output.lines[i],
                                                    additional_attr=output.font_additional_attr)

        return pad, rows, cols

    def _display_nav_bar(self):
        nav_bar_width = self._max_x
        self._nav_bar_pad = _screen_new_output_pad(1, nav_bar_width)
        self._nav_bar = self._nav_history.render(
            nav_bar_width,
            self._NAVIGATION_BACK_COMMAND,
            self._NAVIGATION_FORWARD_COMMAND,
            self._NAVIGATION_HOME_COMMAND,
            self._NAVIGATION_HELP_COMMAND,
            self._NAVIGATION_EXIT_COMMAND)
        self._screen_add_line_to_output_pad(
            self._nav_bar_pad, 0, self._nav_bar.lines[0][:nav_bar_width - 1],
            color_segments=(self._nav_bar.font_attr_segs[0]
                            if 0 in self._nav_bar.font_attr_segs else None),
            additional_attr=self._nav_bar.font_additional_attr)

    def _display_main_menu(self, output):
        """Display main menu associated with screen output, if the menu exists.

        Args:
          output: (ui_common.RichTextLines) The RichTextLines output from
            the annotations field of which the menu will be extracted and used (if
            the menu exists).
        """

        if ui_common.MAIN_MENU_KEY in output.annotations:
            self._main_menu = output.annotations[
                ui_common.MAIN_MENU_KEY].format_as_single_line(
                    prefix="| ", divider=" | ", enabled_item_attrs=["underline"])

            self._main_menu_pad = _screen_new_output_pad(1, self._max_x - 2)

            # The unwrapped menu line may exceed screen width, in which case it needs
            # to be cut off.
            wrapped_menu, _ = ui_common.wrap_rich_text_lines(
                self._main_menu, self._max_x - 3)
            self._screen_add_line_to_output_pad(
                self._main_menu_pad,
                0,
                wrapped_menu.lines[0],
                color_segments=(wrapped_menu.font_attr_segs[0]
                                if 0 in wrapped_menu.font_attr_segs else None),
                additional_attr=wrapped_menu.font_additional_attr)
        else:
            self._main_menu = None
            self._main_menu_pad = None

    def _screen_add_line_to_output_pad(self, pad, row, txt, color_segments=None,
                                       additional_attr=None):
        """Render a line in a text pad.

        Assumes: segments in color_segments are sorted in ascending order of the
        beginning index.
        Note: Gaps between the segments are allowed and will be fixed in with a
        default color.

        Args:
          pad: The text pad to render the line in.
          row: Row index, as an int.
          txt: The text to be displayed on the specified row, as a str.
          color_segments: A list of 3-tuples. Each tuple represents the beginning
            and the end of a color segment, in the form of a right-open interval:
            [start, end). The last element of the tuple is a color string, e.g.,
            "red".

        Raisee:
          TypeError: If color_segments is not of type list.
        """

        if not color_segments:
            pad.addstr(row, 0, txt, self._default_color_pair)
            return

        if not isinstance(color_segments, list):
            raise TypeError("Input color_segments needs to be a list, but is not.")

        all_segments = []
        all_color_pairs = []

        # Process the beginning.
        if color_segments[0][0] == 0:
            pass
        else:
            all_segments.append((0, color_segments[0][0]))
            all_color_pairs.append(self._default_color_pair)

        for (curr_start, curr_end, curr_attrs), (next_start, _, _) in zip(
                color_segments, color_segments[1:] + [(len(txt), None, None)]):
            all_segments.append((curr_start, curr_end))

            if not isinstance(curr_attrs, list):
                curr_attrs = [curr_attrs]

            curses_attr = curses.A_NORMAL
            for attr in curr_attrs:
                if (self._mouse_enabled and
                        isinstance(attr, ui_common.MenuItem)):
                    curses_attr |= curses.A_UNDERLINE
                    if attr.get_custom_color:
                        curses_attr |= self._color_pairs[attr.get_custom_color]
                    else:
                        curses_attr |= self._color_pairs.get(attr, self._default_color_pair)
                else:
                    curses_attr |= self._color_pairs.get(attr, self._default_color_pair)
                if additional_attr:
                    curses_attr |= curses.A_BOLD
                curses_attr |= curses.A_BOLD
            all_color_pairs.append(curses_attr)

            if curr_end < next_start:
                # Fill in the gap with the default color.
                all_segments.append((curr_end, next_start))
                all_color_pairs.append(self._default_color_pair)

        # Finally, draw all the segments.
        for segment, color_pair in zip(all_segments, all_color_pairs):
            if segment[1] < self._max_x:
                pad.addstr(row, segment[0], txt[segment[0]:segment[1]], color_pair)

    def _screen_scroll_output_pad(self, pad, viewport_top, viewport_left,
                                  screen_location_top, screen_location_left,
                                  screen_location_bottom, screen_location_right):
        _refresh_pad(pad, viewport_top, viewport_left, screen_location_top,
                     screen_location_left, screen_location_bottom,
                     screen_location_right)
        self._scroll_bar = ScrollBar(
            self._max_x - 2,
            3 + _TITLE_SIZE_DEFAULT,
            self._max_x - 1,
            self._output_num_rows + _TITLE_SIZE_DEFAULT + 1,
            self._output_pad_row,
            self._output_pad_height - self._output_pad_screen_height)

        (scroll_pad, _, _) = self._display_lines(
            self._scroll_bar.layout(), (self._output_num_rows + _TITLE_SIZE_DEFAULT) - 1)
        _refresh_pad(scroll_pad, 0, 0, self._output_top_row + 1,
                     self._max_x - 2, (self._output_num_rows + _TITLE_SIZE_DEFAULT) + 1,
                     self._max_x - 1)

    def _scroll_output(self, direction, line_index=None):
        """Scroll the output pad.

        Args:
          direction: _SCROLL_REFRESH, _SCROLL_UP, _SCROLL_DOWN, _SCROLL_UP_A_LINE,
            _SCROLL_DOWN_A_LINE, _SCROLL_HOME, _SCROLL_END, _SCROLL_TO_LINE_INDEX
          line_index: (int) Specifies the zero-based line index to scroll to.
            Applicable only if direction is _SCROLL_TO_LINE_INDEX.

        Raises:
          ValueError: On invalid scroll direction.
          TypeError: If line_index is not int and direction is
            _SCROLL_TO_LINE_INDEX.
        """

        if not self._output_pad:
            # No output pad is present. Do nothing.
            return

        if direction == _SCROLL_REFRESH:
            pass
        elif direction == _SCROLL_UP:
            # Scroll up.
            self._output_pad_row -= int(self._output_num_rows / 3)
            if self._output_pad_row < 0:
                self._output_pad_row = 0
        elif direction == _SCROLL_DOWN:
            # Scroll down.
            self._output_pad_row += int(self._output_num_rows / 3)
            if (self._output_pad_row >
                    self._output_pad_height - self._output_pad_screen_height - 1):
                self._output_pad_row = (
                    self._output_pad_height - self._output_pad_screen_height - 1)
        elif direction == _SCROLL_UP_A_LINE:
            # Scroll up a line
            if self._output_pad_row - 1 >= 0:
                self._output_pad_row -= 1
        elif direction == _SCROLL_DOWN_A_LINE:
            # Scroll down a line
            if self._output_pad_row + 1 < (
                    self._output_pad_height - self._output_pad_screen_height):
                self._output_pad_row += 1
        elif direction == _SCROLL_HOME:
            # Scroll to top
            self._output_pad_row = 0
        elif direction == _SCROLL_END:
            # Scroll to bottom
            self._output_pad_row = (
                self._output_pad_height - self._output_pad_screen_height - 1)
        elif direction == _SCROLL_TO_LINE_INDEX:
            if not isinstance(line_index, int):
                raise TypeError("Invalid line_index type (%s) under mode %s" %
                                (type(line_index), _SCROLL_TO_LINE_INDEX))
            self._output_pad_row = line_index
        else:
            raise ValueError("Unsupported scroll mode: %s" % direction)

        self._nav_history.update_scroll_position(self._output_pad_row)

        # Actually scroll the output pad: refresh with new location.
        output_pad_top = self._output_pad_screen_location.top
        if self._main_menu_pad:
            output_pad_top += 1
        self._screen_scroll_output_pad(self._output_pad, self._output_pad_row, 0,
                                       output_pad_top,
                                       self._output_pad_screen_location.left,
                                       self._output_pad_screen_location.bottom,
                                       self._output_pad_screen_location.right)
        self._screen_render_nav_bar()
        self._screen_render_menu_pad()
        self._title()

        self._scroll_info = self._compile_ui_status_summary()
        self._screen_draw_text_line(
            self._output_scroll_row,
            self._scroll_info,
            color=self._STATUS_BAR_COLOR_PAIR)

    def _screen_render_nav_bar(self):
        if self._nav_bar_pad:
            _refresh_pad(self._nav_bar_pad, 0, 0, self._nav_bar_row, 0,
                         self._output_pad_screen_location.top, self._max_x)

    def _screen_render_menu_pad(self):
        if self._main_menu_pad:
            _refresh_pad(
                self._main_menu_pad, 0, 0, self._output_pad_screen_location.top, 0,
                self._output_pad_screen_location.top, self._max_x)

    def _compile_ui_status_summary(self):
        """Compile status summary about this Curses UI instance.

        The information includes: scroll status and mouse ON/OFF status.

        Returns:
          (str) A single text line summarizing the UI status, adapted to the
            current screen width.
        """

        info = ""
        scroll_status_info = ""
        command_info = self._nav_history.get_latest_command_info()
        if self._output_pad_height > self._output_pad_screen_height + 1:
            # Display information about the scrolling of tall screen output.
            scroll_percentage = 100.0 * (min(
                1.0,
                float(self._output_pad_row) /
                (self._output_pad_height - self._output_pad_screen_height - 1)))
            if self._output_pad_row == 0:
                scroll_directions = "(PgDn)"
            elif self._output_pad_row >= (
                    self._output_pad_height - self._output_pad_screen_height - 1):
                scroll_directions = "(PgUp)"
            else:
                scroll_directions = "(PgDn/PgUp)"

            if scroll_percentage < 10.00:
                scroll_percentage = " %.2f%%" % (scroll_percentage)
            else:
                scroll_percentage = "%.2f%%" % (scroll_percentage)
            scroll_status_info = " %s %s" % (scroll_directions, scroll_percentage)

        if len(_COMMAND_TEXT + command_info + scroll_status_info) < self._max_x:
            info = _COMMAND_TEXT + command_info +\
                   " " * (self._max_x - len(_COMMAND_TEXT + command_info + scroll_status_info)) + \
                   scroll_status_info
        else:
            info = _COMMAND_TEXT + command_info[:self._max_x - \
                   len(_COMMAND_TEXT + command_info + scroll_status_info) - 3] + \
                   "..." + scroll_status_info

        return info

    def _tab_complete(self, command_str):
        """Perform tab completion.

        Obtains tab completion candidates.
        If there are no candidates, return command_str and take no other actions.
        If there are candidates, display the candidates on screen and return
        command_str + (common prefix of the candidates).

        Args:
          command_str: (str) The str in the command input textbox when Tab key is
            hit.

        Returns:
          (str) Completed string. Could be the same as command_str if no completion
          candidate is available. If candidate(s) are available, return command_str
          appended by the common prefix of the candidates.
        """

        context, prefix, except_last_word = ui_base._analyze_tab_complete_input(
            command_str)
        candidates, common_prefix = self._tab_completion_registry.get_completions(
            context, prefix)

        if candidates and len(candidates) > 1:
            self._display_candidates(candidates)
        else:
            # In the case of len(candidates) == 1, the single completion will be
            # entered to the textbox automatically. So there is no need to show any
            # candidates.
            self._display_candidates([])

        if common_prefix:
            # Common prefix is not None and non-empty. The completed string will
            # incorporate the common prefix.
            return except_last_word + common_prefix
        return except_last_word + prefix

    def _display_candidates(self, candidates):
        """Show candidates (e.g., tab-completion candidates) on multiple lines.

        Args:
          candidates: (list of str) candidates.
        """

        if self._curr_unwrapped_output:
            # Force refresh screen output.
            self._scroll_output(_SCROLL_REFRESH)

        if not candidates:
            return

        candidates_prefix = "Candidates: "
        candidates_line = candidates_prefix + " ".join(candidates)
        candidates_output = ui_common.RichTextLines(
            candidates_line,
            font_attr_segs={
                0: [(len(candidates_prefix), len(candidates_line), "yellow")]
            })

        candidates_output, _ = ui_common.wrap_rich_text_lines(
            candidates_output, self._max_x - 3)

        # Calculate how many lines the candidate text should occupy. Limit it to
        # a maximum value.
        candidates_num_rows = min(
            len(candidates_output.lines), self._candidates_max_lines)
        self._candidates_top_row = (
            self._candidates_bottom_row - candidates_num_rows + 1)

        # Render the candidate text on screen.
        pad, _, _ = self._display_lines(candidates_output, 0)
        self._screen_scroll_output_pad(
            pad, 0, 0, self._candidates_top_row, 0,
            self._candidates_top_row + candidates_num_rows - 1, self._max_x - 2)

    def _toast(self, message, color=None, line_index=None):
        """Display a one-line message on the screen.

        By default, the toast is displayed in the line right above the scroll bar.
        But the line location can be overridden with the line_index arg.

        Args:
          message: (str) the message to display.
          color: (str) optional color attribute for the message.
          line_index: (int) line index.
        """

        pad, _, _ = self._display_lines(
            ui_common.RichTextLines(
                message,
                font_attr_segs={
                    0: [(0, len(message), color or ui_shared.COLOR_WHITE)]}),
            0)

        right_end = min(len(message), self._max_x - 2)

        if line_index is None:
            line_index = self._output_scroll_row - 1
        self._screen_scroll_output_pad(pad, 0, 0, line_index, 0, line_index,
                                       right_end)

    def _error_toast(self, message):
        """Display a one-line error message on screen.

        Args:
          message: The error message, without the preceding "ERROR: " substring.
        """

        self._toast(
            self.ERROR_MESSAGE_PREFIX + message, color=self._ERROR_TOAST_COLOR_PAIR)

    def _info_toast(self, message):
        """Display a one-line informational message on screen.

        Args:
          message: The informational message.
        """

        self._toast(
            self.INFO_MESSAGE_PREFIX + message, color=self._INFO_TOAST_COLOR_PAIR)

    def _interrupt_handler(self, signal_num, frame):
        del signal_num  # Unused.
        del frame  # Unused.

        if self._on_ui_exit:
            self._on_ui_exit()

        self._screen_terminate()
        print("\ntvmdbg: caught SIGINT; calling sys.exit(1).", file=sys.stderr)
        sys.exit(1)

    def _mouse_mode_command_handler(self, args, screen_info=None):
        """Handler for the command prefix 'mouse'.

        Args:
          args: (list of str) Arguments to the command prefix 'mouse'.
          screen_info: (dict) Information about the screen, unused by this handler.

        Returns:
          None, as this command handler does not generate any screen outputs other
            than toasts.
        """

        del screen_info

        if not args or len(args) == 1:
            if args:
                if args[0].lower() == "on":
                    enabled = True
                elif args[0].lower() == "off":
                    enabled = False
                else:
                    self._error_toast("Invalid mouse mode: %s" % args[0])
                    return None

                self._set_mouse_enabled(enabled)

            mode_str = "on" if self._mouse_enabled else "off"
            self._info_toast("Mouse mode: %s" % mode_str)
        else:
            self._error_toast("mouse_mode: syntax error")

        return None

    def _set_mouse_enabled(self, enabled):
        if self._mouse_enabled != enabled:
            self._mouse_enabled = enabled
            self._screen_set_mousemask()
            self._redraw_output()

    def _screen_set_mousemask(self):
        curses.mousemask(self._mouse_enabled)
