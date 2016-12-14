#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
#  Copyright (C) 1997-2016 JdeRobot Developers Team
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see http://www.gnu.org/licenses/.
#  Authors :
#       Aitor Martinez Fernandez <aitor.martinez.fernandez@gmail.com>
#

import sys
from parallelIce.cameraClient import CameraClient
from gui.threadGUI import ThreadGUI
from gui.GUI import Gui
from PyQt5.QtWidgets import QApplication
import easyiceconfig as EasyIce
import signal

# de componente2
from control.control import Control
from control.threadcontrol import ThreadControl
from control.cameraFilter import CameraFilter

signal.signal(signal.SIGINT, signal.SIG_DFL)

if __name__ == '__main__':

	
	ic = EasyIce.initialize(sys.argv)
	prop = ic.getProperties()
	cameraCli = CameraClient(ic, "basic_component.Camera", True)
	camera = CameraFilter(cameraCli)
	#control = Control(camera)

	app = QApplication(sys.argv)
	frame = Gui()
	frame.setControl(camera)
	frame.show()
	
	t1 = ThreadControl(camera)
	t1.start()
	t2 = ThreadGUI(frame)
	t2.start()

	sys.exit(app.exec_())
