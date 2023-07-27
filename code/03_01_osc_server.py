"""
MaxMSP and AI - Course 03 (OSC)

This script demonstrates the most basic OSC server in Python

@author: esling
"""

from pythonosc import dispatcher, osc_server

def print_filter_freq(unused_addr, freq):
    print(f"Filter frequency: {freq}")

""" Creating the message dispatcher """
dispatch = dispatcher.Dispatcher()
dispatch.map("/filter/frequency", print_filter_freq)
""" Creating the OSC server """
server = osc_server.ThreadingOSCUDPServer(("localhost", 8000), dispatch)
print("Serving on {}".format(server.server_address))
server.serve_forever()
