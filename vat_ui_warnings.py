"""
Simple Warning and error windows to prevent crashes
"""
from PyQt5.QtWidgets import QMessageBox
##
##
##
##
class ErrorMsg(QMessageBox):
    #######################################################################
	# Simple message window to give user a chance to fix erroneous configuration
	# without causing program to crash
    #######################################################################
	def __init__(self, msg, parent=None):
		super(ErrorMsg, self).__init__(parent)
		self.setIcon(QMessageBox.Critical)
		self.setText(msg)
		self.setWindowTitle('Error')
##
##
##
##
class WarningMsg(QMessageBox):
    #######################################################################
	# Simple message window to give user a chance to fix erroneous configuration
	# Catches/informs user when they may proceed in a non-optimal way, but one
	# that would not cause program to crash. May cause bad data, but program will
	# execute.
    #######################################################################
	def __init__(self, msg, parent=None):
		super(WarningMsg, self).__init__(parent)
		self.setText(msg)
		self.setWindowTitle('Warning')
##
##
##
##
class DoneSave(QMessageBox):
    #######################################################################
	# Simple message window to tell user saving is complete
    #######################################################################
	def __init__(self, parent=None):
		super(DoneSave, self).__init__(parent)
		self.setText('Saving Complete')
		self.setWindowTitle('Saving Complete')
