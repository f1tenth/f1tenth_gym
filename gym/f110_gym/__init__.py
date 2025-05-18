import time
import warnings

import gymnasium

gymnasium.register(
	id='f110-v0',
	entry_point='f110_gym.envs:F110Env',
)


class ThrottledPrinter:
	def __init__(self, min_interval=1.0):
		"""
        Create a throttled printer that limits the frequency of messages.

        Args:
            min_interval: Minimum time between prints in seconds
        """
		self.min_interval = min_interval
		self.last_print_time = {}
		# ANSI color codes
		self.colors = {
			'red': '\033[31m',
			'green': '\033[32m',
			'yellow': '\033[33m',
			'blue': '\033[34m',
			'magenta': '\033[35m',
			'cyan': '\033[36m',
			'white': '\033[37m',
			'reset': '\033[0m'
		}

	def print(self, message, color=None):
		"""Print a message if enough time has passed since the last print of this message."""
		current_time = time.time()
		if message not in self.last_print_time or (current_time - self.last_print_time[message]) >= self.min_interval:
			if color and color in self.colors:
				print(f"{self.colors[color]}{message}{self.colors['reset']}")
			else:
				print(message)
			self.last_print_time[message] = current_time

	def warn(self, message, color='yellow'):
		"""Issue a warning if enough time has passed since the last warning with this message."""
		current_time = time.time()
		if message not in self.last_print_time or (current_time - self.last_print_time[message]) >= self.min_interval:
			if color and color in self.colors:
				warnings.warn(f"{self.colors[color]}{message}{self.colors['reset']}")
			else:
				warnings.warn(message)
			self.last_print_time[message] = current_time
