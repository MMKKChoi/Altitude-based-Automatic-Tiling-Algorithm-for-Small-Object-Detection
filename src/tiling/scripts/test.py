#!/usr/bin/env python3

from pymavlink import mavutil

# Create the connection
master = mavutil.mavlink_connection('/dev/ttyACM0', baud=115200)

# Wait for the first heartbeat 
#   This sets the system and component ID of remote system for the link
master.wait_heartbeat()

while True:
    try:
        altitude = master.messages['VFR_HUD'].alt
        print("Altitude: ", altitude)
    except:
        pass

